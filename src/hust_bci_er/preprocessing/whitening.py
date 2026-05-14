"""Channel covariance whitening utilities."""

from __future__ import annotations

import numpy as np


def channel_whiten(x: np.ndarray, eps: float = 1e-3, shrinkage_alpha: float = 0.0) -> np.ndarray:
    """Whiten channels for one EEG window shaped [channels, time]."""
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=1, keepdims=True)
    cov = centered @ centered.T / max(centered.shape[1] - 1, 1)
    if shrinkage_alpha:
        target = np.eye(cov.shape[0]) * (np.trace(cov) / cov.shape[0])
        cov = (1.0 - shrinkage_alpha) * cov + shrinkage_alpha * target
    eigvals, eigvecs = np.linalg.eigh(cov)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, eps))) @ eigvecs.T
    return (inv_sqrt @ centered).astype(np.float32)


def partial_channel_whiten(x: np.ndarray, eps: float = 1e-3, beta: float = 1.0) -> np.ndarray:
    whitened = channel_whiten(x, eps=eps)
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=1, keepdims=True)
    return ((1.0 - beta) * centered + beta * whitened).astype(np.float32)

