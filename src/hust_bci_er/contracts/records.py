"""Dataclasses shared across data, inference, and evaluation modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EEGTrial:
    x: np.ndarray
    y: int | None
    subject_id: str
    trial_id: str
    split: str


@dataclass(frozen=True)
class EEGWindow:
    x: np.ndarray
    y: int | None
    subject_id: str
    trial_id: str
    crop_id: int | None
    window_start_sec: float | None
    split: str


@dataclass(frozen=True)
class PredictionRecord:
    route_id: str
    subject_id: str
    trial_id: str
    crop_id: int | None
    y_score: float
    y_pred: int | None = None
    y_true: int | None = None
    fold: int | None = None
