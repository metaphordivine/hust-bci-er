"""Domain-adversarial training building blocks."""

from __future__ import annotations

import torch
from torch import nn


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, float(lambd))


class DomainClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, n_domains: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_domains),
        )

    def forward(self, x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(x, lambd))

