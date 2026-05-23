"""Channel graph priors for the independent DEP/HC task.

The HUST dataset PDF defines the 30-channel A2-reference order. This module
turns that order into small, explicit graph priors for DEP/HC feature branches.
The graph is a design prior only; it does not encode subject or trial metadata.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from hust_bci_er.features.connectivity import channel_correlation_matrix
from hust_bci_er.models.eeg_montage import (
    HUST_30_A2_CHANNELS,
    HUST_30_A2_REGIONS,
    channel_names_for_montage,
    indices_for_channel_names,
    region_indices_for_montage,
)


HUST_30_A2_REGION_NAMES = ("frontal", "frontocentral", "central", "centroparietal", "posterior")

HUST_30_A2_HOMOLOGOUS_PAIRS: tuple[tuple[str, str], ...] = (
    ("FP1", "FP2"),
    ("F7", "F8"),
    ("F3", "F4"),
    ("FT7", "FT8"),
    ("FC3", "FC4"),
    ("T3", "T4"),
    ("C3", "C4"),
    ("TP7", "TP8"),
    ("CP3", "CP4"),
    ("T5", "T6"),
    ("P3", "P4"),
    ("O1", "O2"),
)


def hust_30_a2_channel_table() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    region_for_channel = _region_lookup()
    for idx, name in enumerate(HUST_30_A2_CHANNELS, start=1):
        rows.append({"index": idx, "name": name, "region": region_for_channel[name]})
    return rows


def graph_edge_indices(
    *,
    channel_montage: str = "hust_30_a2",
    include_region_chain: bool = True,
    include_homologous: bool = True,
    include_self: bool = False,
) -> tuple[tuple[int, int], ...]:
    _require_hust_montage(channel_montage)
    names = channel_names_for_montage(channel_montage, len(HUST_30_A2_CHANNELS))
    edges: set[tuple[int, int]] = set()
    if include_region_chain:
        for region in HUST_30_A2_REGIONS:
            indices = indices_for_channel_names(names, region)
            for left, right in zip(indices, indices[1:]):
                edges.add(_undirected_edge(left, right))
    if include_homologous:
        for left, right in HUST_30_A2_HOMOLOGOUS_PAIRS:
            li, ri = indices_for_channel_names(names, (left, right))
            edges.add(_undirected_edge(li, ri))
    if include_self:
        edges.update((idx, idx) for idx in range(len(names)))
    return tuple(sorted(edges))


def channel_graph_adjacency(
    *,
    channel_montage: str = "hust_30_a2",
    include_region_chain: bool = True,
    include_homologous: bool = True,
    include_self: bool = True,
) -> np.ndarray:
    names = channel_names_for_montage(channel_montage, len(HUST_30_A2_CHANNELS))
    adjacency = np.zeros((len(names), len(names)), dtype=np.float32)
    for left, right in graph_edge_indices(
        channel_montage=channel_montage,
        include_region_chain=include_region_chain,
        include_homologous=include_homologous,
        include_self=include_self,
    ):
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    return adjacency


def regional_connectivity_features(x: np.ndarray, *, channel_montage: str = "hust_30_a2") -> np.ndarray:
    _require_hust_montage(channel_montage)
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("EEG sample must be shaped [channels, time]")
    regions = region_indices_for_montage(
        channel_montage=channel_montage,
        n_channels=arr.shape[0],
        n_regions=len(HUST_30_A2_REGION_NAMES),
    )
    corr = channel_correlation_matrix(arr)
    values: list[float] = []
    for left_idx, left in enumerate(regions):
        for right_idx, right in enumerate(regions[left_idx:], start=left_idx):
            block = corr[np.ix_(list(left), list(right))]
            if left_idx == right_idx:
                block_values = _upper_triangle_without_diagonal(block)
            else:
                block_values = block.reshape(-1)
            if block_values.size == 0:
                values.extend([0.0, 0.0])
            else:
                values.extend([float(np.mean(block_values)), float(np.mean(np.abs(block_values)))])
    return np.asarray(values, dtype=np.float64)


def channel_graph_metadata() -> dict[str, object]:
    return {
        "channel_montage": "hust_30_a2",
        "reference": "A2",
        "sfreq_hz": 250,
        "channels": hust_30_a2_channel_table(),
        "region_names": list(HUST_30_A2_REGION_NAMES),
        "regions": {
            name: list(region)
            for name, region in zip(HUST_30_A2_REGION_NAMES, HUST_30_A2_REGIONS, strict=True)
        },
        "homologous_pairs": [list(pair) for pair in HUST_30_A2_HOMOLOGOUS_PAIRS],
        "edge_count_without_self": len(graph_edge_indices(include_self=False)),
    }


def _region_lookup() -> Mapping[str, str]:
    out: dict[str, str] = {}
    for region_name, channels in zip(HUST_30_A2_REGION_NAMES, HUST_30_A2_REGIONS, strict=True):
        for channel in channels:
            out[channel] = region_name
    return out


def _require_hust_montage(channel_montage: str) -> None:
    names = channel_names_for_montage(channel_montage, len(HUST_30_A2_CHANNELS))
    if tuple(names) != HUST_30_A2_CHANNELS:
        raise ValueError("DEP/HC channel graph priors require the HUST 30-channel A2 montage")


def _undirected_edge(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left <= right else (right, left)


def _upper_triangle_without_diagonal(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return np.asarray([], dtype=np.float64)
    rows, cols = np.triu_indices(min(matrix.shape[0], matrix.shape[1]), k=1)
    return np.asarray(matrix[rows, cols], dtype=np.float64)
