"""Dual-branch graph-conformer EEG classifier."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hust_bci_er.models.eeg_montage import channel_names_for_montage
from hust_bci_er.models.graph.dgcnn import ChebGraphConv
from hust_bci_er.models.graph.lggnet import normalize_adjacency
from hust_bci_er.models.heads.classification import MLPHead


class DualGraphConformer(nn.Module):
    """Channel-graph and channel-time attention fusion for EEG windows."""

    def __init__(
        self,
        *,
        n_channels: int = 30,
        n_times: int = 2500,
        n_classes: int = 2,
        channel_montage: str | None = None,
        embedding_dim: int = 48,
        graph_hidden_dim: int = 32,
        fusion_dim: int = 64,
        k_order: int = 3,
        num_heads: int = 4,
        transformer_depth: int = 1,
        temporal_kernel_size: int = 63,
        token_count: int = 16,
        classifier_hidden_dim: int = 64,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        del n_times
        if n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if embedding_dim <= 0 or graph_hidden_dim <= 0 or fusion_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        if k_order <= 0:
            raise ValueError("k_order must be positive")
        if transformer_depth <= 0:
            raise ValueError("transformer_depth must be positive")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.n_channels = int(n_channels)
        self.token_count = int(token_count)
        self.channel_montage = channel_montage or "sequential"
        self.channel_names = channel_names_for_montage(channel_montage, n_channels)
        self.temporal_stem = nn.Sequential(
            nn.Conv2d(1, embedding_dim, kernel_size=(1, temporal_kernel_size), padding=(0, temporal_kernel_size // 2), bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((n_channels, token_count)),
            nn.Dropout(dropout),
        )
        self.channel_pos = nn.Parameter(torch.zeros(1, n_channels, 1, embedding_dim))
        self.time_pos = nn.Parameter(torch.zeros(1, 1, token_count, embedding_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

        self.adjacency_logits = nn.Parameter(torch.zeros(n_channels, n_channels))
        self.graph_conv = ChebGraphConv(embedding_dim, graph_hidden_dim, k_order=k_order)
        self.graph_norm = nn.LayerNorm(graph_hidden_dim)
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_hidden_dim * n_channels, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ELU(),
        )
        self.attention_proj = nn.Sequential(
            nn.Linear(embedding_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ELU(),
        )
        self.fusion_gate = nn.Linear(fusion_dim * 2, fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = MLPHead(
            fusion_dim,
            n_classes=n_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

    def adjacency(self) -> torch.Tensor:
        sym_logits = 0.5 * (self.adjacency_logits + self.adjacency_logits.t())
        return normalize_adjacency(F.softplus(sym_logits))

    def stem_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("DualGraphConformer input must be shaped [batch, channels, time] or [batch, 1, channels, time]")
        return self.temporal_stem(x)

    def graph_features(self, stem: torch.Tensor) -> torch.Tensor:
        nodes = stem.mean(dim=-1).transpose(1, 2)
        graph = self.graph_conv(nodes, self.adjacency())
        graph = self.dropout(F.elu(self.graph_norm(graph)))
        return self.graph_proj(graph.flatten(start_dim=1))

    def attention_features(self, stem: torch.Tensor) -> torch.Tensor:
        tokens = stem.permute(0, 2, 3, 1)
        tokens = tokens + self.channel_pos + self.time_pos
        tokens = tokens.reshape(tokens.shape[0], self.n_channels * self.token_count, -1)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        encoded = self.transformer(torch.cat([cls, tokens], dim=1))
        return self.attention_proj(encoded[:, 0])

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem_features(x)
        graph = self.graph_features(stem)
        attention = self.attention_features(stem)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([graph, attention], dim=-1)))
        fused = gate * graph + (1.0 - gate) * attention
        return self.dropout(fused)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


def build_dual_graph_conformer(n_channels: int, n_times: int, n_classes: int = 2, **kwargs) -> DualGraphConformer:
    return DualGraphConformer(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        channel_montage=kwargs.get("channel_montage"),
        embedding_dim=int(kwargs.get("embedding_dim", 48)),
        graph_hidden_dim=int(kwargs.get("graph_hidden_dim", 32)),
        fusion_dim=int(kwargs.get("fusion_dim", 64)),
        k_order=int(kwargs.get("k_order", 3)),
        num_heads=int(kwargs.get("num_heads", 4)),
        transformer_depth=int(kwargs.get("transformer_depth", 1)),
        temporal_kernel_size=int(kwargs.get("temporal_kernel_size", 63)),
        token_count=int(kwargs.get("token_count", 16)),
        classifier_hidden_dim=int(kwargs.get("classifier_hidden_dim", 64)),
        dropout=float(kwargs.get("dropout", 0.35)),
    )
