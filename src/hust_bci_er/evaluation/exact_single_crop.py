"""Exact single-crop expected BA for 8 trials x 5 crops."""

from __future__ import annotations

import numpy as np

from hust_bci_er.inference.topk import topk_binary

N_TRIALS = 8
N_CROPS = 5


def assignment_grid(n_trials: int = N_TRIALS, n_crops: int = N_CROPS) -> np.ndarray:
    n_assignments = n_crops**n_trials
    ids = np.arange(n_assignments, dtype=np.int64)
    out = np.empty((n_assignments, n_trials), dtype=np.int16)
    for col in range(n_trials):
        out[:, col] = (ids // (n_crops**col)) % n_crops
    return out


def top4_predictions(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        return topk_binary(scores, 4).astype(np.int8)
    bias = np.arange(scores.shape[1], dtype=np.float64) * 1e-12
    idx = np.argsort(-(scores - bias), axis=1, kind="mergesort")[:, :4]
    pred = np.zeros_like(scores, dtype=np.int8)
    pred[np.arange(scores.shape[0])[:, None], idx] = 1
    return pred


def exact_ba_from_matrix(mat: np.ndarray, y_true: np.ndarray, grid: np.ndarray | None = None) -> float:
    mat = np.asarray(mat, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int8)
    grid = assignment_grid(mat.shape[0], mat.shape[1]) if grid is None else grid
    selected = mat[np.arange(mat.shape[0]), grid]
    pred = top4_predictions(selected)
    return float((pred == y_true[None, :]).mean(axis=1).mean())


def exact_all_correct_rate_from_matrix(mat: np.ndarray, y_true: np.ndarray, grid: np.ndarray | None = None) -> float:
    mat = np.asarray(mat, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int8)
    grid = assignment_grid(mat.shape[0], mat.shape[1]) if grid is None else grid
    selected = mat[np.arange(mat.shape[0]), grid]
    pred = top4_predictions(selected)
    return float((pred == y_true[None, :]).all(axis=1).mean())

