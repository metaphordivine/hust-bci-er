"""FBCNet-style filter-bank spatial filtering classifier.

The model keeps the auditable core of FBCNet for this repository: parallel
temporal filter-bank responses, per-band spatial filters, and log-variance
features over temporal segments.
"""

from __future__ import annotations

import torch
from torch import nn

from hust_bci_er.models.heads.classification import MLPHead


def segment_log_variance(x: torch.Tensor, n_segments: int = 4, eps: float = 1e-6) -> torch.Tensor:
    """Compute log-variance features over equal temporal segments."""

    if x.ndim != 4:
        raise ValueError("segment_log_variance input must be shaped [batch, filters, 1, time]")
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")
    n_times = x.shape[-1]
    if n_times < n_segments:
        raise ValueError("time dimension must be >= n_segments")
    usable = (n_times // n_segments) * n_segments
    x = x[..., :usable].reshape(x.shape[0], x.shape[1], x.shape[2], n_segments, usable // n_segments)
    var = x.var(dim=-1, unbiased=False).clamp_min(eps)
    return torch.log(var).flatten(start_dim=1)


class FBCNet(nn.Module):
    """Compact FBCNet-style network for raw EEG windows."""

    def __init__(
        self,
        *,
        n_channels: int = 30,
        n_times: int = 2500,
        n_classes: int = 2,
        n_bands: int = 9,
        spatial_filters: int = 4,
        temporal_kernel_size: int = 63,
        n_segments: int = 4,
        classifier_hidden_dim: int = 64,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        del n_times
        if n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if n_bands <= 0:
            raise ValueError("n_bands must be positive")
        if spatial_filters <= 0:
            raise ValueError("spatial_filters must be positive")
        if temporal_kernel_size <= 0:
            raise ValueError("temporal_kernel_size must be positive")
        if n_segments <= 0:
            raise ValueError("n_segments must be positive")
        self.n_segments = int(n_segments)
        self.temporal_filterbank = nn.Sequential(
            nn.Conv2d(1, n_bands, kernel_size=(1, temporal_kernel_size), padding=(0, temporal_kernel_size // 2), bias=False),
            nn.BatchNorm2d(n_bands),
            nn.ELU(),
        )
        self.spatial_filters = nn.Sequential(
            nn.Conv2d(
                n_bands,
                n_bands * spatial_filters,
                kernel_size=(n_channels, 1),
                groups=n_bands,
                bias=False,
            ),
            nn.BatchNorm2d(n_bands * spatial_filters),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        feature_dim = n_bands * spatial_filters * n_segments
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
            raise ValueError("FBCNet input must be shaped [batch, channels, time] or [batch, 1, channels, time]")
        x = self.temporal_filterbank(x)
        x = self.spatial_filters(x)
        return segment_log_variance(x, n_segments=self.n_segments)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


def build_fbcnet(n_channels: int, n_times: int, n_classes: int = 2, **kwargs) -> FBCNet:
    return FBCNet(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        n_bands=int(kwargs.get("n_bands", 9)),
        spatial_filters=int(kwargs.get("spatial_filters", 4)),
        temporal_kernel_size=int(kwargs.get("temporal_kernel_size", 63)),
        n_segments=int(kwargs.get("n_segments", 4)),
        classifier_hidden_dim=int(kwargs.get("classifier_hidden_dim", 64)),
        dropout=float(kwargs.get("dropout", 0.4)),
    )
