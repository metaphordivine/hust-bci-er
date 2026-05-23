"""Train-fold-only preprocessing statistics for EEG windows."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def _window_arrays(windows: Sequence[Mapping[str, Any]]) -> list[np.ndarray]:
    arrays = [np.asarray(item["x"], dtype=np.float64) for item in windows]
    if not arrays:
        raise ValueError("at least one training window is required to fit preprocessing statistics")
    n_channels = arrays[0].shape[0]
    for arr in arrays:
        if arr.ndim != 2:
            raise ValueError("EEG windows must be shaped [channels, time]")
        if arr.shape[0] != n_channels:
            raise ValueError("all EEG windows must have the same channel count")
    return arrays


def _stack_by_channel(windows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    arrays = _window_arrays(windows)
    return np.concatenate(arrays, axis=1)


def fit_channel_normalization(
    windows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Fit per-channel normalization stats from training windows only."""
    data = _stack_by_channel(windows)
    if method == "mean_std":
        center = data.mean(axis=1)
        scale = data.std(axis=1)
    elif method == "median_MAD":
        center = np.median(data, axis=1)
        mad = np.median(np.abs(data - center[:, None]), axis=1)
        scale = 1.4826 * mad
    else:
        raise ValueError(f"unsupported train channel normalization method: {method}")
    scale = np.maximum(scale, float(eps))
    return {
        "schema_version": 1,
        "source": "train_fold_only",
        "scope": "per_channel",
        "method": method,
        "eps": float(eps),
        "center": center.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
    }


def apply_channel_normalization(
    windows: Sequence[dict[str, Any]],
    stats: Mapping[str, Any],
) -> list[dict[str, Any]]:
    center = np.asarray(stats.get("center"), dtype=np.float32)[:, None]
    scale = np.asarray(stats.get("scale"), dtype=np.float32)[:, None]
    if center.ndim != 2 or scale.ndim != 2 or center.shape != scale.shape:
        raise ValueError("channel normalization stats must contain matching center/scale vectors")
    for item in windows:
        x = np.asarray(item["x"], dtype=np.float32)
        if x.shape[0] != center.shape[0]:
            raise ValueError("channel normalization stats channel count does not match window")
        item["x"] = ((x - center) / scale).astype(np.float32)
    return list(windows)


def fit_robust_clip(
    windows: Sequence[Mapping[str, Any]],
    *,
    clip_n_mad: float,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Fit per-channel robust clipping bounds from training windows only."""
    if clip_n_mad <= 0:
        raise ValueError("clip_n_mad must be positive")
    data = _stack_by_channel(windows)
    center = np.median(data, axis=1)
    mad = np.median(np.abs(data - center[:, None]), axis=1)
    mad = np.maximum(mad, float(eps))
    lower = center - float(clip_n_mad) * mad
    upper = center + float(clip_n_mad) * mad
    return {
        "schema_version": 1,
        "source": "train_fold_only",
        "scope": "per_channel",
        "method": "per_channel_robust_clip",
        "center": "median",
        "scale": "MAD",
        "clip_n_mad": float(clip_n_mad),
        "eps": float(eps),
        "median": center.astype(float).tolist(),
        "mad": mad.astype(float).tolist(),
        "lower": lower.astype(float).tolist(),
        "upper": upper.astype(float).tolist(),
    }


def apply_robust_clip(
    windows: Sequence[dict[str, Any]],
    stats: Mapping[str, Any],
    *,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lower = np.asarray(stats.get("lower"), dtype=np.float32)[:, None]
    upper = np.asarray(stats.get("upper"), dtype=np.float32)[:, None]
    if lower.ndim != 2 or upper.ndim != 2 or lower.shape != upper.shape:
        raise ValueError("robust clip stats must contain matching lower/upper vectors")

    n_values = 0
    n_clipped = 0
    n_windows = 0
    n_clipped_windows = 0
    by_channel = np.zeros(lower.shape[0], dtype=np.int64)
    by_subject: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_cohort: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_label: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_trial: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for item in windows:
        x = np.asarray(item["x"], dtype=np.float32)
        if x.shape[0] != lower.shape[0]:
            raise ValueError("robust clip stats channel count does not match window")
        mask = (x < lower) | (x > upper)
        clipped = int(mask.sum())
        values = int(x.size)
        n_values += values
        n_clipped += clipped
        n_windows += 1
        clipped_window = int(clipped > 0)
        n_clipped_windows += clipped_window
        by_channel += mask.sum(axis=1)
        subject = str(item.get("subject_id", ""))
        cohort = str(item.get("cohort", ""))
        label = str(item.get("y", ""))
        trial = str(item.get("trial_id", ""))
        for store, key in ((by_subject, subject), (by_cohort, cohort), (by_label, label), (by_trial, trial)):
            store[key][0] += clipped_window
            store[key][1] += 1
        item["x"] = np.clip(x, lower, upper).astype(np.float32)

    def rates(raw: Mapping[str, list[int]]) -> dict[str, float]:
        return {key: (count / total if total else 0.0) for key, (count, total) in sorted(raw.items())}

    audit = {
        "split": str(split),
        "n_windows": int(n_windows),
        "n_values": int(n_values),
        "n_clipped_values": int(n_clipped),
        "clip_value_fraction": float(n_clipped / n_values) if n_values else 0.0,
        "clipped_window_fraction": float(n_clipped_windows / n_windows) if n_windows else 0.0,
        "clip_rate_by_channel": (by_channel / max(1, n_windows * int(windows[0]["x"].shape[1]) if windows else 1)).astype(float).tolist(),
        "clip_window_rate_by_subject": rates(by_subject),
        "clip_window_rate_by_cohort": rates(by_cohort),
        "clip_window_rate_by_label": rates(by_label),
        "clip_window_rate_by_trial": rates(by_trial),
    }
    return list(windows), audit
