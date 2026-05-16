"""Compact LGGNet-style local/global graph EEG classifier.

This is a repo-native implementation of the registered ``lggnet`` graph model.
It follows the published LGGNet design at the component level: multi-scale
temporal convolutions, kernel-level fusion, local region aggregation, and a
learnable global graph filter over brain regions.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from hust_bci_er.models.eeg_montage import contiguous_region_indices, even_region_indices, region_indices_for_montage
from hust_bci_er.models.heads.classification import MLPHead


def even_region_sizes(n_channels: int, n_regions: int) -> tuple[int, ...]:
    return tuple(len(indices) for indices in even_region_indices(n_channels, n_regions))


class TemporalKernelFusion(nn.Module):
    """Multi-scale temporal convolution with learnable branch weights."""

    def __init__(
        self,
        *,
        n_filters: int,
        kernel_sizes: Sequence[int],
        pool_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_filters <= 0:
            raise ValueError("n_filters must be positive")
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty")
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        branches = []
        for kernel_size in kernel_sizes:
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError("temporal kernel sizes must be positive odd integers")
            branches.append(
                nn.Sequential(
                    nn.Conv2d(1, n_filters, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2), bias=False),
                    nn.BatchNorm2d(n_filters),
                    nn.ELU(),
                    nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_size)),
                    nn.Dropout(dropout),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.branch_logits = nn.Parameter(torch.zeros(len(branches)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("LGGNet input must be shaped [batch, channels, time] or [batch, 1, channels, time]")
        outputs = torch.stack([branch(x) for branch in self.branches], dim=1)
        weights = torch.softmax(self.branch_logits, dim=0).view(1, -1, 1, 1, 1)
        return (outputs * weights).sum(dim=1)


class RegionAggregator(nn.Module):
    """Aggregate channel features into configured local brain regions."""

    def __init__(self, region_indices: Sequence[Sequence[int]]) -> None:
        super().__init__()
        if not region_indices:
            raise ValueError("region_indices must not be empty")
        parsed = tuple(tuple(int(idx) for idx in region) for region in region_indices)
        if any(not region for region in parsed):
            raise ValueError("region_indices must contain non-empty regions")
        flat = [idx for region in parsed for idx in region]
        if min(flat) < 0:
            raise ValueError("region indices must be non-negative")
        if len(set(flat)) != len(flat):
            raise ValueError("region indices must be disjoint")
        self.region_indices = parsed

    @property
    def n_regions(self) -> int:
        return len(self.region_indices)

    @property
    def n_channels(self) -> int:
        return max(max(region) for region in self.region_indices) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("region aggregation input must be shaped [batch, channels, features]")
        if x.shape[1] <= max(max(region) for region in self.region_indices):
            raise ValueError(f"expected at least {self.n_channels} channels, got {x.shape[1]}")
        pooled = []
        for region in self.region_indices:
            idx = torch.as_tensor(region, device=x.device, dtype=torch.long)
            pooled.append(x.index_select(1, idx).mean(dim=1))
        return torch.stack(pooled, dim=1)


class GraphConvolution(nn.Module):
    """Dense graph convolution over region nodes."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("graph dimensions must be positive")
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("graph input must be shaped [batch, nodes, features]")
        support = self.proj(x)
        return torch.matmul(adjacency, support) + self.bias


def normalize_adjacency(adjacency: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    adjacency = adjacency.clamp_min(0.0)
    adjacency = adjacency + torch.eye(adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype)
    degree = adjacency.sum(dim=1).clamp_min(eps)
    inv_sqrt = torch.rsqrt(degree)
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


class LGGNet(nn.Module):
    """Local-global graph network for EEG windows."""

    def __init__(
        self,
        *,
        n_channels: int = 30,
        n_times: int = 2500,
        n_classes: int = 2,
        n_regions: int = 5,
        region_sizes: Sequence[int] | None = None,
        region_indices: Sequence[Sequence[int]] | None = None,
        temporal_filters: int = 8,
        temporal_kernel_sizes: Sequence[int] = (31, 63, 125),
        temporal_pool_size: int = 4,
        graph_hidden_dim: int = 32,
        classifier_hidden_dim: int = 64,
        dropout: float = 0.3,
        channel_montage: str | None = None,
    ) -> None:
        super().__init__()
        del n_times
        if region_indices is not None and region_sizes is not None:
            raise ValueError("declare either region_indices or region_sizes, not both")
        if region_indices is None:
            if region_sizes is not None:
                if sum(int(size) for size in region_sizes) != n_channels:
                    raise ValueError("sum(region_sizes) must equal n_channels")
                region_indices = contiguous_region_indices(region_sizes)
            else:
                region_indices = region_indices_for_montage(
                    channel_montage=channel_montage,
                    n_channels=n_channels,
                    n_regions=n_regions,
                )
        if graph_hidden_dim <= 0:
            raise ValueError("graph_hidden_dim must be positive")
        self.n_channels = int(n_channels)
        self.channel_montage = channel_montage or "sequential"
        self.region_indices = tuple(tuple(int(idx) for idx in region) for region in region_indices)
        self.temporal = TemporalKernelFusion(
            n_filters=temporal_filters,
            kernel_sizes=temporal_kernel_sizes,
            pool_size=temporal_pool_size,
            dropout=dropout,
        )
        self.region_aggregator = RegionAggregator(self.region_indices)
        self.adjacency_logits = nn.Parameter(torch.zeros(len(self.region_indices), len(self.region_indices)))
        self.graph_conv = GraphConvolution(temporal_filters, graph_hidden_dim)
        self.node_norm = nn.LayerNorm(graph_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = MLPHead(
            graph_hidden_dim * len(self.region_indices),
            n_classes=n_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

    def region_features(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal(x)
        # Log-power summaries are stable across different temporal crop lengths.
        power = torch.log(temporal.pow(2).mean(dim=-1).clamp_min(1e-6))
        # [B, C, n_filters] so RegionAggregator pools over the channel axis.
        channel_features = power.transpose(1, 2)
        return self.region_aggregator(channel_features)

    def adjacency(self) -> torch.Tensor:
        sym_logits = 0.5 * (self.adjacency_logits + self.adjacency_logits.t())
        return normalize_adjacency(F.softplus(sym_logits))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        nodes = self.region_features(x)
        graph = self.graph_conv(nodes, self.adjacency())
        graph = self.dropout(F.elu(self.node_norm(graph)))
        return graph.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


def build_lggnet(n_channels: int, n_times: int, n_classes: int = 2, **kwargs) -> LGGNet:
    return LGGNet(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        n_regions=int(kwargs.get("n_regions", 5)),
        region_sizes=kwargs.get("region_sizes"),
        region_indices=kwargs.get("region_indices"),
        temporal_filters=int(kwargs.get("temporal_filters", 8)),
        temporal_kernel_sizes=tuple(kwargs.get("temporal_kernel_sizes", (31, 63, 125))),
        temporal_pool_size=int(kwargs.get("temporal_pool_size", 4)),
        graph_hidden_dim=int(kwargs.get("graph_hidden_dim", 32)),
        classifier_hidden_dim=int(kwargs.get("classifier_hidden_dim", 64)),
        dropout=float(kwargs.get("dropout", 0.3)),
        channel_montage=kwargs.get("channel_montage"),
    )
