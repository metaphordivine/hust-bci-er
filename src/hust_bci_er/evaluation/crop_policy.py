"""Crop policy utilities for pseudo-public and stress-test evaluation."""

from __future__ import annotations

import numpy as np


def select_crop_matrix(mat: np.ndarray, policy: str, random_state: int | None = None) -> np.ndarray:
    """Return a [trials, 1] matrix for fixed or random crop policies.

    `worst` is label-aware and intentionally not implemented here. It belongs
    in stress-test evaluation code, not real inference.
    """
    mat = np.asarray(mat, dtype=np.float64)
    if policy.startswith("crop"):
        idx = int(policy.replace("crop", "")) - 1
        return mat[:, idx : idx + 1]
    if policy == "random":
        rng = np.random.default_rng(random_state)
        idx = rng.integers(0, mat.shape[1], size=mat.shape[0])
        return mat[np.arange(mat.shape[0]), idx][:, None]
    raise ValueError(f"unsupported non-label crop policy: {policy}")


def worst_crop_score(scores_by_assignment: np.ndarray, y_true: np.ndarray, metric_fn) -> tuple[float, int]:
    """Return worst assignment score and index for a label-aware stress test."""
    values = np.asarray([metric_fn(row, y_true) for row in scores_by_assignment], dtype=np.float64)
    idx = int(np.argmin(values))
    return float(values[idx]), idx

