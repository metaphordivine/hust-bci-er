"""Compact EEG Conformer backbone."""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class EEGConformerLite(nn.Module):
    def __init__(
        self,
        n_channels: int = 30,
        n_times: int = 2500,
        n_classes: int = 2,
        emb_size: int = 32,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.5,
        conv_dropout: float = 0.3,
    ):
        super().__init__()
        del n_times
        self.conv_stem = nn.Sequential(
            nn.Conv2d(1, emb_size // 2, (1, 32), padding=(0, 16)),
            nn.BatchNorm2d(emb_size // 2),
            nn.ELU(),
            nn.Conv2d(emb_size // 2, emb_size, (n_channels, 1)),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 32)),
            nn.Dropout(conv_dropout),
        )
        self.pos_enc = PositionalEncoding(emb_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=emb_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.classifier = nn.Sequential(nn.LayerNorm(emb_size), nn.Linear(emb_size, n_classes))

    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv_stem(x).squeeze(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        return self.transformer(self.pos_enc(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_tokens(x)[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


def build_conformer_lite(n_channels: int, n_times: int, n_classes: int = 2, **kwargs) -> EEGConformerLite:
    return EEGConformerLite(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        emb_size=kwargs.get("emb_size", 32),
        depth=kwargs.get("depth", 2),
        num_heads=kwargs.get("num_heads", 4),
        dropout=kwargs.get("dropout", 0.5),
        conv_dropout=kwargs.get("conv_dropout", 0.3),
    )

