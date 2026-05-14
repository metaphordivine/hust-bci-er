"""Euclidean alignment helpers.

Fit alignment only on the training split, then apply the fitted matrix to
validation/test/public data. Do not fit EA after splitting windows randomly.
"""

from __future__ import annotations

import numpy as np


def covariance(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=1, keepdims=True)
    cov = centered @ centered.T / max(centered.shape[1] - 1, 1)
    return cov + np.eye(cov.shape[0]) * eps


def fit_ea_transform(windows: list[np.ndarray], eps: float = 1e-6) -> np.ndarray:
    if not windows:
        raise ValueError("at least one training window is required")
    mean_cov = np.mean([covariance(x, eps=eps) for x in windows], axis=0)
    eigvals, eigvecs = np.linalg.eigh(mean_cov)
    return (eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, eps))) @ eigvecs.T).astype(np.float32)


def apply_ea_transform(x: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (np.asarray(transform) @ np.asarray(x)).astype(np.float32)

