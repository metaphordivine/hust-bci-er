"""Clean EEGNet implementation without trained parameters."""

from __future__ import annotations

import torch
from torch import nn


class EEGNet(nn.Module):
    def __init__(
        self,
        *,
        n_channels: int,
        n_classes: int = 2,
        dropout: float = 0.5,
        kernel_length: int = 64,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        return_features: bool = False,
    ) -> None:
        super().__init__()
        self.return_features = return_features
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(f1, f1 * d, kernel_size=(n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, kernel_size=(1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(f2, n_classes),
        )
        self._feature_dim = f2

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor, return_features: bool | None = None) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.first_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        use_features = self.return_features if return_features is None else return_features
        if use_features:
            return self.head[1](self.head[0](x))
        return self.head(x)


def build_eegnet(
    n_channels: int,
    n_times: int,
    n_classes: int = 2,
    **kwargs,
) -> EEGNet:
    del n_times
    return EEGNet(
        n_channels=n_channels,
        n_classes=n_classes,
        dropout=kwargs.get("dropout", 0.5),
        kernel_length=kwargs.get("kernel_length", 64),
        f1=kwargs.get("f1", kwargs.get("F1", 8)),
        d=kwargs.get("d", kwargs.get("D", 2)),
        f2=kwargs.get("f2", kwargs.get("F2", 16)),
        return_features=kwargs.get("return_features", False),
    )

