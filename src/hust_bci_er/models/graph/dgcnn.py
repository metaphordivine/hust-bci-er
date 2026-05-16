"""DGCNN-style dynamic graph convolution for EEG windows."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hust_bci_er.models.heads.classification import MLPHead
from hust_bci_er.models.graph.lggnet import normalize_adjacency


class ChebGraphConv(nn.Module):
    """Chebyshev graph convolution with a dense normalized adjacency."""

    def __init__(self, input_dim: int, output_dim: int, k_order: int = 3) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("graph dimensions must be positive")
        if k_order <= 0:
            raise ValueError("k_order must be positive")
        self.k_order = int(k_order)
        self.projections = nn.ModuleList([nn.Linear(input_dim, output_dim, bias=False) for _ in range(k_order)])
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("graph input must be shaped [batch, nodes, features]")
        supports = [x]
        if self.k_order > 1:
            supports.append(torch.matmul(adjacency, x))
        for _ in range(2, self.k_order):
            supports.append(2.0 * torch.matmul(adjacency, supports[-1]) - supports[-2])
        out = sum(proj(support) for proj, support in zip(self.projections, supports))
        return out + self.bias


class DGCNN(nn.Module):
    """Dynamic graph CNN over channel nodes."""

    def __init__(
        self,
        *,
        n_channels: int = 30,
        n_times: int = 2500,
        n_classes: int = 2,
        node_features: int = 16,
        graph_hidden_dim: int = 32,
        k_order: int = 3,
        temporal_kernel_size: int = 63,
        classifier_hidden_dim: int = 64,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        del n_times
        if n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if node_features <= 0 or graph_hidden_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        self.n_channels = int(n_channels)
        self.temporal_encoder = nn.Sequential(
            nn.Conv2d(1, node_features, kernel_size=(1, temporal_kernel_size), padding=(0, temporal_kernel_size // 2), bias=False),
            nn.BatchNorm2d(node_features),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((n_channels, 1)),
        )
        self.adjacency_logits = nn.Parameter(torch.zeros(n_channels, n_channels))
        self.graph_conv = ChebGraphConv(node_features, graph_hidden_dim, k_order=k_order)
        self.node_norm = nn.LayerNorm(graph_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = MLPHead(
            graph_hidden_dim * n_channels,
            n_classes=n_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

    def adjacency(self) -> torch.Tensor:
        sym_logits = 0.5 * (self.adjacency_logits + self.adjacency_logits.t())
        return normalize_adjacency(F.softplus(sym_logits))

    def node_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("DGCNN input must be shaped [batch, channels, time] or [batch, 1, channels, time]")
        encoded = self.temporal_encoder(x).squeeze(-1)
        return encoded.transpose(1, 2)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        nodes = self.node_embeddings(x)
        graph = self.graph_conv(nodes, self.adjacency())
        graph = self.dropout(F.elu(self.node_norm(graph)))
        return graph.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


def build_dgcnn(n_channels: int, n_times: int, n_classes: int = 2, **kwargs) -> DGCNN:
    return DGCNN(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        node_features=int(kwargs.get("node_features", 16)),
        graph_hidden_dim=int(kwargs.get("graph_hidden_dim", 32)),
        k_order=int(kwargs.get("k_order", 3)),
        temporal_kernel_size=int(kwargs.get("temporal_kernel_size", 63)),
        classifier_hidden_dim=int(kwargs.get("classifier_hidden_dim", 64)),
        dropout=float(kwargs.get("dropout", 0.4)),
    )
