"""Hjorth time-domain features."""

from __future__ import annotations

import numpy as np


def hjorth_features(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    dx = np.diff(signal, axis=-1)
    ddx = np.diff(dx, axis=-1)
    var0 = np.var(signal, axis=-1)
    var1 = np.var(dx, axis=-1)
    var2 = np.var(ddx, axis=-1)
    activity = var0
    mobility = np.sqrt(var1 / np.maximum(var0, eps))
    complexity = np.sqrt(var2 / np.maximum(var1, eps)) / np.maximum(mobility, eps)
    return np.stack([activity, mobility, complexity], axis=1).astype(np.float32)

