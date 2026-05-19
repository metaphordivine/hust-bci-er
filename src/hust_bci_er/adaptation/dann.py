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


class DomainAdversarialClassifier(nn.Module):
    """Wrap a feature-extractor classifier with a DANN domain head."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        feature_dim: int,
        domain_hidden_dim: int = 64,
        n_domains: int = 2,
        domain_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not hasattr(base_model, "extract_features") or not hasattr(base_model, "classifier"):
            raise ValueError("DANN adaptation requires a model with extract_features() and classifier")
        self.base_model = base_model
        self.domain_classifier = DomainClassifier(
            input_dim=int(feature_dim),
            hidden_dim=int(domain_hidden_dim),
            n_domains=int(n_domains),
            dropout=float(domain_dropout),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model.extract_features(x)
        if features.ndim != 2:
            raise ValueError("DANN extract_features output must be shaped [batch, features]")
        return features

    def label_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.base_model.classifier(features)

    def domain_logits_from_features(self, features: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        return self.domain_classifier(features, lambd=lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.label_logits_from_features(self.extract_features(x))
