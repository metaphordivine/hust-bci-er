"""Reusable classifier heads for trainable EEG models."""

from __future__ import annotations

import torch
from torch import nn


def _validate_head_args(input_dim: int, n_classes: int, dropout: float) -> None:
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")


class LinearHead(nn.Module):
    """LayerNorm + optional dropout + linear classifier."""

    def __init__(self, input_dim: int, n_classes: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        _validate_head_args(input_dim, n_classes, dropout)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("classifier head input must be shaped [batch, features]")
        return self.net(x)


class MLPHead(nn.Module):
    """Small MLP classifier used by compact graph/backbone models."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int = 2,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        _validate_head_args(input_dim, n_classes, dropout)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("classifier head input must be shaped [batch, features]")
        return self.net(x)
