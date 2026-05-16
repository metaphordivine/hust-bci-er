"""TSception-style temporal/spatial convolutional EEG classifier.

The implementation keeps the TSception idea that matters for this repository:
multi-scale temporal kernels followed by global and hemispheric spatial
summaries. It is intentionally compact so it can run through the existing
``torch_classifier`` adapter and model smoke gate.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from hust_bci_er.models.heads.classification import MLPHead


class TSceptionBranch(nn.Module):
    def __init__(
        self,
        *,
        n_channels: int,
        n_filters: int,
        kernel_size: int,
        temporal_pool_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_channels < 2:
            raise ValueError("TSception requires at least two channels for hemispheric spatial branches")
        if n_filters <= 0:
            raise ValueError("n_filters must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if temporal_pool_size <= 0:
            raise ValueError("temporal_pool_size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.left_channels = n_channels // 2
        self.right_channels = n_channels - self.left_channels
        self.temporal = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2), bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, temporal_pool_size), stride=(1, temporal_pool_size)),
            nn.Dropout(dropout),
        )
        self.global_spatial = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
        )
        self.left_spatial = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel_size=(self.left_channels, 1), bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
        )
        self.right_spatial = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel_size=(self.right_channels, 1), bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def _pooled(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x).flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        left = x[:, :, : self.left_channels, :]
        right = x[:, :, self.left_channels :, :]
        return torch.cat(
            [
                self._pooled(self.global_spatial(x)),
                self._pooled(self.left_spatial(left)),
                self._pooled(self.right_spatial(right)),
            ],
            dim=1,
        )


class TSception(nn.Module):
    """Compact TSception-style classifier for EEG windows."""

    def __init__(
        self,
        *,
        n_channels: int = 30,
        n_times: int = 2500,
        n_classes: int = 2,
        n_filters: int = 8,
        temporal_kernel_sizes: Sequence[int] | None = None,
        temporal_pool_size: int = 8,
        classifier_hidden_dim: int = 64,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if temporal_kernel_sizes is None:
            temporal_kernel_sizes = tuple(max(3, int(round(n_times * frac)) | 1) for frac in (0.5, 0.25, 0.125))
        if not temporal_kernel_sizes:
            raise ValueError("temporal_kernel_sizes must not be empty")
        self.branches = nn.ModuleList(
            [
                TSceptionBranch(
                    n_channels=n_channels,
                    n_filters=n_filters,
                    kernel_size=int(kernel),
                    temporal_pool_size=temporal_pool_size,
                    dropout=dropout,
                )
                for kernel in temporal_kernel_sizes
            ]
        )
        feature_dim = len(self.branches) * n_filters * 3
        self.classifier = MLPHead(
            feature_dim,
            n_classes=n_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("TSception input must be shaped [batch, channels, time] or [batch, 1, channels, time]")
        return torch.cat([branch(x) for branch in self.branches], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


def build_tsception(n_channels: int, n_times: int, n_classes: int = 2, **kwargs) -> TSception:
    return TSception(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        n_filters=int(kwargs.get("n_filters", 8)),
        temporal_kernel_sizes=kwargs.get("temporal_kernel_sizes"),
        temporal_pool_size=int(kwargs.get("temporal_pool_size", 8)),
        classifier_hidden_dim=int(kwargs.get("classifier_hidden_dim", 64)),
        dropout=float(kwargs.get("dropout", 0.4)),
    )
