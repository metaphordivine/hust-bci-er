"""SRFNet-style raw plus frequency fusion backbone."""

from __future__ import annotations

import math

import torch
from torch import nn

from .eegnet import EEGNet
from .fft import FFTFeatureLayer


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class RawConformerBranch(nn.Module):
    def __init__(
        self,
        *,
        n_channels: int,
        n_times: int,
        emb_size: int = 32,
        depth: int = 2,
        num_heads: int = 4,
        patch_size: int = 32,
        dropout: float = 0.3,
        conv_dropout: float = 0.18,
        feature_dim: int = 64,
    ) -> None:
        super().__init__()
        stem_width = max(emb_size // 2, 8)
        token_count = max(16, min(64, n_times // max(int(patch_size), 1)))
        self.conv_stem = nn.Sequential(
            nn.Conv2d(1, stem_width, kernel_size=(1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ELU(),
            nn.Conv2d(stem_width, emb_size, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, token_count)),
            nn.Dropout(conv_dropout),
        )
        self.pos_enc = PositionalEncoding(emb_size, max_len=token_count + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=emb_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.feature_proj = nn.Sequential(nn.LayerNorm(emb_size), nn.Linear(emb_size, feature_dim), nn.GELU(), nn.Dropout(dropout))
        self.score_head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        tokens = self.conv_stem(x).squeeze(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(tokens.shape[0], -1, -1)
        encoded = self.transformer(self.pos_enc(torch.cat([cls_tokens, tokens], dim=1)))
        z_conf = self.feature_proj(encoded[:, 0])
        return z_conf, self.score_head(z_conf)


class FftEEGNetBranch(nn.Module):
    def __init__(
        self,
        *,
        n_channels: int,
        dropout: float = 0.35,
        kernel_length: int = 64,
        f1: int = 16,
        d: int = 2,
        f2: int = 32,
        fft_mode: str = "log_power",
        normalize_fft: bool = False,
        feature_dim: int = 64,
    ) -> None:
        super().__init__()
        self.fft = FFTFeatureLayer(mode=fft_mode, normalize=normalize_fft)
        self.encoder = EEGNet(
            n_channels=n_channels,
            n_classes=2,
            dropout=dropout,
            kernel_length=kernel_length,
            f1=f1,
            d=d,
            f2=f2,
            return_features=True,
        )
        self.feature_proj = nn.Sequential(nn.Linear(self.encoder.feature_dim, feature_dim), nn.BatchNorm1d(feature_dim), nn.GELU(), nn.Dropout(dropout))
        self.score_head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_fft = self.encoder(self.fft(x), return_features=True)
        z_fft = self.feature_proj(z_fft)
        return z_fft, self.score_head(z_fft)


class SRFNet(nn.Module):
    def __init__(
        self,
        *,
        n_channels: int = 30,
        n_times: int = 2500,
        emb_size: int = 32,
        depth: int = 2,
        num_heads: int = 4,
        patch_size: int = 32,
        dropout: float = 0.3,
        feature_dim: int = 64,
        fft_dropout: float = 0.35,
        fusion_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.raw_branch = RawConformerBranch(
            n_channels=n_channels,
            n_times=n_times,
            emb_size=emb_size,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            dropout=dropout,
            conv_dropout=max(dropout * 0.6, 0.12),
            feature_dim=feature_dim,
        )
        self.fft_branch = FftEEGNetBranch(n_channels=n_channels, dropout=fft_dropout, feature_dim=feature_dim)
        self.gate = nn.Sequential(nn.Linear(feature_dim * 2, fusion_hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(fusion_hidden_dim, 1))
        self.residual_head = nn.Sequential(
            nn.Linear(feature_dim * 2 + 2, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_conf, score_conf = self.raw_branch(x)
        z_fft, score_fft = self.fft_branch(x)
        gate_input = torch.cat([z_conf, z_fft], dim=1)
        fusion_input = torch.cat([z_conf, z_fft, score_conf, score_fft], dim=1)
        gate = torch.sigmoid(self.gate(gate_input))
        residual = self.residual_head(fusion_input)
        score = gate * score_conf + (1.0 - gate) * score_fft + residual
        return {
            "score": score.squeeze(-1),
            "score_conf": score_conf.squeeze(-1),
            "score_fft": score_fft.squeeze(-1),
            "z_conf": z_conf,
            "z_fft": z_fft,
            "gate": gate.squeeze(-1),
            "residual": residual.squeeze(-1),
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.forward_features(x)
        score = outputs["score"]
        outputs["logits"] = torch.stack([-score, score], dim=1)
        return outputs


def build_srfnet(n_channels: int, n_times: int, n_classes: int = 2, **kwargs) -> SRFNet:
    if n_classes != 2:
        raise ValueError("SRFNet score formulation currently expects binary classification")
    return SRFNet(n_channels=n_channels, n_times=n_times, **kwargs)

