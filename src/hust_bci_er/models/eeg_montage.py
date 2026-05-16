"""EEG channel montage helpers for model components.

The HUST 30-channel order is taken from the local dataset PDF. These helpers
keep channel-order assumptions explicit for models that use spatial, regional,
or graph structure.
"""

from __future__ import annotations

from collections.abc import Sequence


HUST_30_A2_CHANNELS = (
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "FT7",
    "FC3",
    "FCZ",
    "FC4",
    "FT8",
    "T3",
    "C3",
    "CZ",
    "C4",
    "T4",
    "TP7",
    "CP3",
    "CPZ",
    "CP4",
    "TP8",
    "T5",
    "P3",
    "PZ",
    "P4",
    "T6",
    "O1",
    "OZ",
    "O2",
)

HUST_30_A2_LEFT = ("FP1", "F7", "F3", "FT7", "FC3", "T3", "C3", "TP7", "CP3", "T5", "P3", "O1")
HUST_30_A2_RIGHT = ("FP2", "F4", "F8", "FC4", "FT8", "C4", "T4", "CP4", "TP8", "P4", "T6", "O2")
HUST_30_A2_REGIONS = (
    ("FP1", "FP2", "F7", "F3", "FZ", "F4", "F8"),
    ("FT7", "FC3", "FCZ", "FC4", "FT8"),
    ("T3", "C3", "CZ", "C4", "T4"),
    ("TP7", "CP3", "CPZ", "CP4", "TP8"),
    ("T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2"),
)

SUPPORTED_MONTAGES = {"hust_30_a2": HUST_30_A2_CHANNELS}


def channel_names_for_montage(channel_montage: str | None, n_channels: int) -> tuple[str, ...]:
    if n_channels <= 0:
        raise ValueError("n_channels must be positive")
    if channel_montage in {None, "", "sequential"}:
        return tuple(f"CH{idx + 1}" for idx in range(n_channels))
    if channel_montage not in SUPPORTED_MONTAGES:
        raise ValueError(f"unknown channel_montage: {channel_montage}")
    names = SUPPORTED_MONTAGES[str(channel_montage)]
    if len(names) != n_channels:
        raise ValueError(f"channel_montage {channel_montage} expects {len(names)} channels, got {n_channels}")
    return names


def indices_for_channel_names(channel_names: Sequence[str], selected: Sequence[str]) -> tuple[int, ...]:
    lookup = {name.upper(): idx for idx, name in enumerate(channel_names)}
    try:
        return tuple(lookup[name.upper()] for name in selected)
    except KeyError as exc:
        raise ValueError(f"channel montage is missing expected channel: {exc.args[0]}") from exc


def contiguous_region_indices(region_sizes: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    if not region_sizes or any(int(size) <= 0 for size in region_sizes):
        raise ValueError("region_sizes must contain positive integers")
    regions = []
    start = 0
    for size in region_sizes:
        stop = start + int(size)
        regions.append(tuple(range(start, stop)))
        start = stop
    return tuple(regions)


def even_region_indices(n_channels: int, n_regions: int) -> tuple[tuple[int, ...], ...]:
    if n_regions <= 0:
        raise ValueError("n_regions must be positive")
    if n_regions > n_channels:
        raise ValueError("n_regions must not exceed n_channels")
    base, extra = divmod(n_channels, n_regions)
    sizes = tuple(base + (1 if idx < extra else 0) for idx in range(n_regions))
    return contiguous_region_indices(sizes)


def hemisphere_indices(
    *,
    channel_montage: str | None,
    n_channels: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    names = channel_names_for_montage(channel_montage, n_channels)
    if channel_montage == "hust_30_a2":
        return (
            indices_for_channel_names(names, HUST_30_A2_LEFT),
            indices_for_channel_names(names, HUST_30_A2_RIGHT),
        )
    left_n = n_channels // 2
    return tuple(range(left_n)), tuple(range(left_n, n_channels))


def region_indices_for_montage(
    *,
    channel_montage: str | None,
    n_channels: int,
    n_regions: int,
) -> tuple[tuple[int, ...], ...]:
    names = channel_names_for_montage(channel_montage, n_channels)
    if channel_montage == "hust_30_a2" and n_regions == len(HUST_30_A2_REGIONS):
        return tuple(indices_for_channel_names(names, region) for region in HUST_30_A2_REGIONS)
    return even_region_indices(n_channels, n_regions)
