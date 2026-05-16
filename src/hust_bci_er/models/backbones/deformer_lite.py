"""Compact EEG Deformer-style backbone."""

from __future__ import annotations

import math

import torch
from torch import nn


class EEGDeformerLite(nn.Module):
    def __init__(
        self,
        n_channels: int = 30,
        n_times: int = 2500,
        n_classes: int = 2,
        conv_channels: int = 16,
        transformer_depth: int = 1,
        num_heads: int = 4,
        embedding_dim: int = 32,
        dropout: float = 0.4,
        feedforward_multiplier: int = 2,
        pooling: str = "mean",
        classifier_hidden_dim: int | None = None,
    ):
        super().__init__()
        del n_times
        if pooling not in {"mean", "attention"}:
            raise ValueError(f"pooling must be mean or attention, got {pooling}")
        self.pooling = pooling
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv_channels, (1, 64), padding=(0, 32)),
            nn.BatchNorm2d(conv_channels),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout * 0.5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_channels, conv_channels * 2, (n_channels, 1)),
            nn.BatchNorm2d(conv_channels * 2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout * 0.5),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv_channels * 2, embedding_dim, (1, 32), padding=(0, 16)),
            nn.BatchNorm2d(embedding_dim),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Dropout(dropout),
        )
        pos = torch.zeros(2000, embedding_dim)
        position = torch.arange(0, 2000).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pos[:, 0::2] = torch.sin(position * div_term)
        pos[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_enc", pos.unsqueeze(0))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * feedforward_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)
        self.attention_pool = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 1),
        ) if pooling == "attention" else None
        hidden_dim = int(classifier_hidden_dim or max(embedding_dim // 2, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x).squeeze(2).transpose(1, 2)
        x = x + self.pos_enc[:, : x.shape[1], :]
        x = self.transformer(x)
        if self.attention_pool is not None:
            weights = torch.softmax(self.attention_pool(x).squeeze(-1), dim=1)
            x = torch.sum(x * weights.unsqueeze(-1), dim=1)
        else:
            x = x.mean(dim=1)
        return self.classifier(x)


def build_deformer_lite(n_channels: int, n_times: int, n_classes: int = 2, **kwargs) -> EEGDeformerLite:
    classifier_hidden_dim = kwargs.get("classifier_hidden_dim")
    return EEGDeformerLite(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        conv_channels=kwargs.get("conv_channels", 16),
        transformer_depth=kwargs.get("transformer_depth", 1),
        num_heads=kwargs.get("num_heads", 4),
        embedding_dim=kwargs.get("embedding_dim", 32),
        dropout=kwargs.get("dropout", 0.4),
        feedforward_multiplier=int(kwargs.get("feedforward_multiplier", 2)),
        pooling=str(kwargs.get("pooling", "mean")),
        classifier_hidden_dim=int(classifier_hidden_dim) if classifier_hidden_dim is not None else None,
    )

