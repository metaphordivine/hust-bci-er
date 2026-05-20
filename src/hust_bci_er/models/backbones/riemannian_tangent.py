"""Riemannian tangent-space covariance classifier for EEG windows."""

from __future__ import annotations

import math

import torch
from torch import nn


class RiemannianTangentNet(nn.Module):
    """Log-covariance tangent features followed by a linear classifier.

    This keeps the route in the existing torch_classifier pipeline while making
    the model itself a shallow covariance/tangent-space baseline rather than a
    learned temporal convolutional network.
    """

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        covariance_eps: float = 1e-3,
        shrinkage: float = 0.1,
        tangent_scale: float = math.sqrt(2.0),
        feature_standardize: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_chans < 2:
            raise ValueError("RiemannianTangentNet requires at least two channels")
        if n_times < 2:
            raise ValueError("RiemannianTangentNet requires at least two time samples")
        if covariance_eps <= 0:
            raise ValueError("covariance_eps must be positive")
        if not 0.0 <= shrinkage < 1.0:
            raise ValueError("shrinkage must be in [0, 1)")
        if tangent_scale <= 0:
            raise ValueError("tangent_scale must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.n_chans = int(n_chans)
        self.n_outputs = int(n_outputs)
        self.n_times = int(n_times)
        self.covariance_eps = float(covariance_eps)
        self.shrinkage = float(shrinkage)
        self.tangent_scale = float(tangent_scale)
        self.feature_dim = self.n_chans * (self.n_chans + 1) // 2

        rows, cols = torch.triu_indices(self.n_chans, self.n_chans)
        self.register_buffer("triu_rows", rows, persistent=False)
        self.register_buffer("triu_cols", cols, persistent=False)
        self.register_buffer("offdiag_mask", (rows != cols), persistent=False)

        self.feature_norm = nn.LayerNorm(self.feature_dim, elementwise_affine=False) if feature_standardize else nn.Identity()
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(self.feature_dim, self.n_outputs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError("4D EEG input must be shaped [batch, 1, channels, times]")
            x = x.squeeze(1)
        if x.ndim != 3:
            raise ValueError("EEG input must be shaped [batch, channels, times]")
        if x.shape[1] != self.n_chans:
            raise ValueError(f"expected {self.n_chans} channels, got {x.shape[1]}")
        if x.shape[2] < 2:
            raise ValueError("EEG input must contain at least two time samples")
        return x

    def covariance(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        centered = x - x.mean(dim=-1, keepdim=True)
        denom = max(int(centered.shape[-1]) - 1, 1)
        cov = centered @ centered.transpose(-1, -2) / float(denom)

        eye = torch.eye(self.n_chans, dtype=cov.dtype, device=cov.device).expand(cov.shape[0], -1, -1)
        trace_mean = cov.diagonal(dim1=-2, dim2=-1).mean(dim=-1).view(-1, 1, 1)
        cov = (1.0 - self.shrinkage) * cov + self.shrinkage * trace_mean * eye
        cov = cov + self.covariance_eps * eye
        return cov

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        cov = self.covariance(x)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        log_diag = torch.log(torch.clamp(eigvals, min=self.covariance_eps))
        log_cov = (eigvecs * log_diag.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
        features = log_cov[:, self.triu_rows, self.triu_cols]
        features = torch.where(self.offdiag_mask.unsqueeze(0), features * self.tangent_scale, features)
        return self.feature_norm(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.classifier(self.dropout(features))


def build_riemannian_tangent(
    n_channels: int,
    n_times: int,
    n_classes: int = 2,
    **kwargs: object,
) -> RiemannianTangentNet:
    return RiemannianTangentNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        covariance_eps=float(kwargs.get("covariance_eps", 1e-3)),
        shrinkage=float(kwargs.get("shrinkage", 0.1)),
        tangent_scale=float(kwargs.get("tangent_scale", math.sqrt(2.0))),
        feature_standardize=bool(kwargs.get("feature_standardize", True)),
        dropout=float(kwargs.get("dropout", 0.0)),
    )
