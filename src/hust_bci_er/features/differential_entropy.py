"""Differential entropy features for Gaussian band signals."""

from __future__ import annotations

import math

import numpy as np


def differential_entropy_from_variance(var: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    var = np.maximum(np.asarray(var, dtype=np.float64), eps)
    return (0.5 * np.log(2.0 * math.pi * math.e * var)).astype(np.float32)


def differential_entropy_features(signal: np.ndarray, sfreq: float, bands: list[tuple[float, float]] | None = None) -> np.ndarray:
    from .bandpower import DEFAULT_BANDS

    signal = np.asarray(signal, dtype=np.float64)
    bands = bands or DEFAULT_BANDS
    freqs = np.fft.rfftfreq(signal.shape[-1], d=1.0 / sfreq)
    fft = np.fft.rfft(signal, axis=-1)
    feats = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if not mask.any():
            feats.append(np.zeros(signal.shape[0], dtype=np.float32))
            continue
        band_power = np.abs(fft[:, mask]) ** 2
        feats.append(differential_entropy_from_variance(band_power.mean(axis=1)))
    return np.stack(feats, axis=1).astype(np.float32)

