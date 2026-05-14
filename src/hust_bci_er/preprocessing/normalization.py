"""Normalization helpers for EEG arrays shaped [channels, time]."""

from __future__ import annotations

import numpy as np


def zscore_per_channel(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return ((x - mean) / np.maximum(std, eps)).astype(np.float32)


def robust_zscore_per_channel(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    median = np.median(x, axis=-1, keepdims=True)
    mad = np.median(np.abs(x - median), axis=-1, keepdims=True)
    scale = 1.4826 * mad
    return ((x - median) / np.maximum(scale, eps)).astype(np.float32)


def common_average_reference(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean(axis=0, keepdims=True)).astype(np.float32)

