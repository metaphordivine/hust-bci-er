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


def query_context_fusion(query: np.ndarray, context_scores: list[np.ndarray], *, alpha: float, temperature: float) -> np.ndarray:
    q = zscore_rows(query)
    context_items = [zscore_rows(v) for v in context_scores]
    sims = np.stack([(q * v).mean(axis=1) for v in context_items], axis=1) / float(temperature)
    sims = sims - sims.max(axis=1, keepdims=True)
    weights = np.exp(sims)
    weights = weights / weights.sum(axis=1, keepdims=True)
    context = sum(weights[:, i : i + 1] * context_items[i] for i in range(len(context_items)))
    return (1.0 - float(alpha)) * q + float(alpha) * context
