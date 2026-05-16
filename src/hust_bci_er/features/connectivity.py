"""Lightweight channel-connectivity feature extraction."""

from __future__ import annotations

import numpy as np


def channel_correlation_matrix(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return a Pearson channel-correlation matrix for ``[channels, time]`` input."""

    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("signal must be shaped [channels, time]")
    centered = x - x.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.sum(centered * centered, axis=1, keepdims=True)).clip(min=eps)
    normalized = centered / scale
    corr = normalized @ normalized.T
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32)


def channel_covariance_matrix(signal: np.ndarray) -> np.ndarray:
    """Return a channel covariance matrix for ``[channels, time]`` input."""

    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("signal must be shaped [channels, time]")
    if x.shape[1] < 2:
        return np.zeros((x.shape[0], x.shape[0]), dtype=np.float32)
    centered = x - x.mean(axis=1, keepdims=True)
    cov = centered @ centered.T / (x.shape[1] - 1)
    return cov.astype(np.float32)


def upper_triangle_features(matrix: np.ndarray, *, include_diagonal: bool = False) -> np.ndarray:
    """Flatten the upper triangle of a square matrix."""

    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be square")
    offset = 0 if include_diagonal else 1
    rows, cols = np.triu_indices(mat.shape[0], k=offset)
    return mat[rows, cols].astype(np.float32)


def connectivity_features(
    signal: np.ndarray,
    *,
    method: str = "pearson",
    include_diagonal: bool = False,
) -> np.ndarray:
    """Return pairwise channel-connectivity features.

    ``subject_id`` and trial identifiers are intentionally not accepted here;
    this function only consumes numeric EEG windows.
    """

    if method == "pearson":
        matrix = channel_correlation_matrix(signal)
    elif method == "abs_pearson":
        matrix = np.abs(channel_correlation_matrix(signal))
    elif method == "covariance":
        matrix = channel_covariance_matrix(signal)
    else:
        raise ValueError("method must be one of: pearson, abs_pearson, covariance")
    return upper_triangle_features(matrix, include_diagonal=include_diagonal)
