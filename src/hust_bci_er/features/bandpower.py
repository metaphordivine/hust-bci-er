"""Bandpower-style features for hand-crafted routes."""

from __future__ import annotations

import numpy as np


DEFAULT_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def bandpower_features(signal: np.ndarray, sfreq: float, bands: list[tuple[float, float]] | None = None) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    bands = bands or DEFAULT_BANDS
    freqs = np.fft.rfftfreq(signal.shape[-1], d=1.0 / sfreq)
    power = np.abs(np.fft.rfft(signal, axis=-1)) ** 2
    feats = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        value = np.log1p(power[:, mask].mean(axis=1)) if mask.any() else np.zeros(signal.shape[0])
        feats.append(value)
    return np.stack(feats, axis=1).astype(np.float32)

