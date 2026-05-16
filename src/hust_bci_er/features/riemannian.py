"""Riemannian covariance and tangent-space feature utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def regularized_covariance(x: np.ndarray, eps: float = 1e-6, shrinkage_alpha: float = 0.0) -> np.ndarray:
    """Return a symmetric positive definite channel covariance matrix."""

    signal = np.asarray(x, dtype=np.float64)
    if signal.ndim != 2:
        raise ValueError("signal must be shaped [channels, time]")
    centered = signal - signal.mean(axis=1, keepdims=True)
    denom = max(signal.shape[1] - 1, 1)
    cov = centered @ centered.T / denom
    if shrinkage_alpha:
        if not 0.0 <= shrinkage_alpha <= 1.0:
            raise ValueError("shrinkage_alpha must be in [0, 1]")
        target = np.eye(cov.shape[0]) * (np.trace(cov) / cov.shape[0])
        cov = (1.0 - shrinkage_alpha) * cov + shrinkage_alpha * target
    cov = 0.5 * (cov + cov.T)
    cov = cov + np.eye(cov.shape[0]) * eps
    return cov.astype(np.float64)


def matrix_log_spd(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Matrix logarithm for a symmetric positive definite matrix."""

    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be square")
    eigvals, eigvecs = np.linalg.eigh(0.5 * (mat + mat.T))
    log_vals = np.log(np.maximum(eigvals, eps))
    return (eigvecs * log_vals) @ eigvecs.T


def inv_sqrtm_spd(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Inverse square root for a symmetric positive definite matrix."""

    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be square")
    eigvals, eigvecs = np.linalg.eigh(0.5 * (mat + mat.T))
    inv_sqrt = 1.0 / np.sqrt(np.maximum(eigvals, eps))
    return (eigvecs * inv_sqrt) @ eigvecs.T


def tangent_vector(matrix: np.ndarray, *, include_diagonal: bool = True) -> np.ndarray:
    """Vectorize a symmetric tangent matrix with sqrt(2) off-diagonal scaling."""

    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be square")
    offset = 0 if include_diagonal else 1
    rows, cols = np.triu_indices(mat.shape[0], k=offset)
    values = mat[rows, cols].copy()
    # Off-diagonal sqrt(2) scaling preserves the Frobenius inner product after
    # symmetric-matrix vectorization.
    off_diag = rows != cols
    values[off_diag] *= np.sqrt(2.0)
    return values.astype(np.float32)


def tangent_space_features(
    signal: np.ndarray,
    *,
    reference_covariance: np.ndarray | None = None,
    eps: float = 1e-6,
    shrinkage_alpha: float = 0.0,
) -> np.ndarray:
    """Project one EEG window covariance into a tangent-space feature vector."""

    cov = regularized_covariance(signal, eps=eps, shrinkage_alpha=shrinkage_alpha)
    if reference_covariance is not None:
        inv_ref = inv_sqrtm_spd(reference_covariance, eps=eps)
        cov = inv_ref @ cov @ inv_ref
    return tangent_vector(matrix_log_spd(cov, eps=eps))


@dataclass
class TangentSpaceProjector:
    """Fit a log-Euclidean reference covariance and transform EEG windows."""

    eps: float = 1e-6
    shrinkage_alpha: float = 0.0
    reference_covariance_: np.ndarray | None = None

    def fit(self, windows: list[np.ndarray] | tuple[np.ndarray, ...]) -> "TangentSpaceProjector":
        if not windows:
            raise ValueError("windows must not be empty")
        covariances = [regularized_covariance(x, eps=self.eps, shrinkage_alpha=self.shrinkage_alpha) for x in windows]
        log_mean = np.mean([matrix_log_spd(cov, eps=self.eps) for cov in covariances], axis=0)
        eigvals, eigvecs = np.linalg.eigh(0.5 * (log_mean + log_mean.T))
        self.reference_covariance_ = (eigvecs * np.exp(eigvals)) @ eigvecs.T
        return self

    def transform(self, windows: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
        if self.reference_covariance_ is None:
            raise ValueError("TangentSpaceProjector must be fit before transform")
        return np.vstack(
            [
                tangent_space_features(
                    x,
                    reference_covariance=self.reference_covariance_,
                    eps=self.eps,
                    shrinkage_alpha=self.shrinkage_alpha,
                )
                for x in windows
            ]
        ).astype(np.float32)

    def fit_transform(self, windows: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
        return self.fit(windows).transform(windows)
