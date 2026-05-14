"""Top-k inference policies."""

from __future__ import annotations

import numpy as np


def topk_binary(scores: np.ndarray, k: int) -> np.ndarray:
    """Return a binary vector with the k largest scores marked as 1."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not 0 <= k <= scores.size:
        raise ValueError("k must be between 0 and len(scores)")
    order = np.argsort(-scores, kind="mergesort")
    pred = np.zeros(scores.size, dtype=int)
    pred[order[:k]] = 1
    return pred


def subject_top4(scores: np.ndarray) -> np.ndarray:
    """BCI ER task policy: mark the top 4 of 8 subject trials as positive."""
    scores = np.asarray(scores, dtype=float)
    if scores.size != 8:
        raise ValueError("subject_top4 expects exactly 8 trial scores")
    return topk_binary(scores, 4)

