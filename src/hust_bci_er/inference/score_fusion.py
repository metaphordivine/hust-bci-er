"""Score-level fusion helpers for route assembly."""

from __future__ import annotations

import numpy as np


def zscore_rows(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / np.maximum(sd, eps)


def z_average(*arrays: np.ndarray, weights: list[float] | None = None) -> np.ndarray:
    if not arrays:
        raise ValueError("at least one score array is required")
    weights = weights or [1.0 / len(arrays)] * len(arrays)
    if len(weights) != len(arrays):
        raise ValueError("weights length must match arrays")
    out = np.zeros_like(np.asarray(arrays[0], dtype=np.float64))
    for weight, arr in zip(weights, arrays):
        out += float(weight) * zscore_rows(arr)
    return out


def qkv_source_fusion(query: np.ndarray, key_values: list[np.ndarray], *, alpha: float, temperature: float) -> np.ndarray:
    q = zscore_rows(query)
    kv = [zscore_rows(v) for v in key_values]
    sims = np.stack([(q * v).mean(axis=1) for v in kv], axis=1) / float(temperature)
    sims = sims - sims.max(axis=1, keepdims=True)
    weights = np.exp(sims)
    weights = weights / weights.sum(axis=1, keepdims=True)
    context = sum(weights[:, i : i + 1] * kv[i] for i in range(len(kv)))
    return (1.0 - float(alpha)) * q + float(alpha) * context

