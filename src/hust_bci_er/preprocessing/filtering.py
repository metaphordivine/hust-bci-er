"""Signal filtering helpers for EEG preprocessing."""

from __future__ import annotations

import warnings

import numpy as np


def butter_bandpass_sos(
    *,
    low_hz: float,
    high_hz: float,
    sfreq: float,
    order: int = 4,
) -> np.ndarray:
    """Design a Butterworth bandpass filter in second-order-section form."""

    if sfreq <= 0:
        raise ValueError("sfreq must be positive")
    if order <= 0:
        raise ValueError("order must be positive")
    nyquist = sfreq / 2.0
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError("bandpass cutoffs must satisfy 0 < low_hz < high_hz < sfreq/2")
    from scipy.signal import butter

    return butter(order, (float(low_hz), float(high_hz)), btype="bandpass", fs=float(sfreq), output="sos")


def _safe_padlen(sos: np.ndarray, n_times: int) -> int | None:
    default = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
    return 0 if n_times <= default else None


def bandpass_filter(
    x: np.ndarray,
    *,
    sfreq: float,
    low_hz: float = 1.0,
    high_hz: float = 45.0,
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filtering along the time axis.

    Very short signals use ``padlen=0`` to keep scipy from failing; callers
    should treat that path as a degraded boundary-quality fallback.
    """

    signal = np.asarray(x, dtype=np.float64)
    sos = butter_bandpass_sos(low_hz=low_hz, high_hz=high_hz, sfreq=sfreq, order=order)
    from scipy.signal import sosfiltfilt

    n_times = signal.shape[axis]
    padlen = _safe_padlen(sos, n_times)
    if padlen == 0:
        warnings.warn(
            "bandpass_filter used padlen=0 because the signal is shorter than scipy's default padding; "
            "boundary quality is degraded for this diagnostic-length window.",
            RuntimeWarning,
            stacklevel=2,
        )
    filtered = sosfiltfilt(sos, signal, axis=axis, padlen=padlen)
    return filtered.astype(np.float32)
