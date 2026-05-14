"""Metric implementations used by audits and reports."""

from __future__ import annotations

import numpy as np


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    classes = np.unique(y_true)
    recalls = []
    for cls in classes:
        mask = y_true == cls
        recalls.append(float((y_pred[mask] == cls).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def all_correct_rate(y_true_matrix: np.ndarray, y_pred_matrix: np.ndarray) -> float:
    y_true_matrix = np.asarray(y_true_matrix, dtype=int)
    y_pred_matrix = np.asarray(y_pred_matrix, dtype=int)
    if y_true_matrix.shape != y_pred_matrix.shape:
        raise ValueError("matrices must have the same shape")
    if y_true_matrix.ndim != 2:
        raise ValueError("matrices must be two-dimensional")
    return float((y_true_matrix == y_pred_matrix).all(axis=1).mean())

