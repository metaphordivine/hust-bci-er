"""Feature extractors for the independent DEP/HC classification task.

These functions consume numeric EEG windows only. Subject and trial identifiers
remain split/audit metadata and are intentionally not accepted as inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hust_bci_er.analysis.dep_hc_router import DEFAULT_BANDS, bandpower_features, covariance_tangent_features
from hust_bci_er.features.connectivity import connectivity_features
from hust_bci_er.models.eeg_montage import channel_names_for_montage, hemisphere_indices, indices_for_channel_names


DEP_HC_FEATURE_SETS = frozenset(
    {
        "cov_tangent",
        "bandpower",
        "cov_tangent_bandpower",
        "traditional",
        "connectivity",
        "time_frequency",
        "traditional_time_frequency",
    }
)

FRONTAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("FP1", "FP2"),
    ("F3", "F4"),
    ("F7", "F8"),
    ("FC3", "FC4"),
)


def dep_hc_features(
    x: np.ndarray,
    *,
    feature_set: str = "traditional",
    sfreq: float = 250.0,
    channel_montage: str | None = "hust_30_a2",
) -> np.ndarray:
    if feature_set == "cov_tangent":
        return covariance_tangent_features(x)
    if feature_set == "bandpower":
        return bandpower_features(x, sfreq=sfreq)
    if feature_set == "cov_tangent_bandpower":
        return np.concatenate([covariance_tangent_features(x), bandpower_features(x, sfreq=sfreq)])
    if feature_set == "connectivity":
        return connectivity_summary_features(x)
    if feature_set == "time_frequency":
        return time_frequency_summary_features(x, sfreq=sfreq)
    if feature_set == "traditional":
        return np.concatenate(
            [
                covariance_tangent_features(x),
                bandpower_features(x, sfreq=sfreq),
                connectivity_summary_features(x),
                asymmetry_features(x, sfreq=sfreq, channel_montage=channel_montage),
            ]
        )
    if feature_set == "traditional_time_frequency":
        return np.concatenate(
            [
                dep_hc_features(x, feature_set="traditional", sfreq=sfreq, channel_montage=channel_montage),
                time_frequency_summary_features(x, sfreq=sfreq),
            ]
        )
    raise ValueError(f"unknown DEP/HC task feature_set: {feature_set}")


def connectivity_summary_features(x: np.ndarray) -> np.ndarray:
    pearson = connectivity_features(x, method="pearson")
    abs_pearson = connectivity_features(x, method="abs_pearson")
    return np.concatenate([pearson, abs_pearson]).astype(np.float64, copy=False)


def asymmetry_features(
    x: np.ndarray,
    *,
    sfreq: float = 250.0,
    channel_montage: str | None = "hust_30_a2",
    bands: Sequence[tuple[str, float, float]] = DEFAULT_BANDS,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("EEG sample must be shaped [channels, time]")
    if channel_montage != "hust_30_a2":
        raise ValueError("DEP/HC asymmetry features require channel_montage='hust_30_a2'")
    names = channel_names_for_montage(channel_montage, arr.shape[0])
    powers = _band_log_power_by_channel(arr, sfreq=sfreq, bands=bands)
    band_lookup = {name: idx for idx, (name, _low, _high) in enumerate(bands)}
    alpha_idx = band_lookup["alpha"]
    beta_idx = band_lookup["beta"]
    left_idx, right_idx = hemisphere_indices(channel_montage=channel_montage, n_channels=arr.shape[0])

    values: list[float] = []
    for left, right in FRONTAL_PAIRS:
        li, ri = indices_for_channel_names(names, (left, right))
        values.append(float(powers[ri, alpha_idx] - powers[li, alpha_idx]))
        values.append(float(powers[ri, beta_idx] - powers[li, beta_idx]))
    for band_idx in range(len(bands)):
        values.append(float(np.mean(powers[list(right_idx), band_idx]) - np.mean(powers[list(left_idx), band_idx])))
    return np.asarray(values, dtype=np.float64)


def time_frequency_summary_features(
    x: np.ndarray,
    *,
    sfreq: float = 250.0,
    bands: Sequence[tuple[str, float, float]] = DEFAULT_BANDS,
    n_segments: int = 5,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("EEG sample must be shaped [channels, time]")
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")
    chunks = [chunk for chunk in np.array_split(arr, int(n_segments), axis=1) if chunk.shape[1] >= 2]
    if not chunks:
        raise ValueError("EEG sample must contain at least two time samples")
    segment_band = np.stack(
        [_band_log_power_by_channel(chunk, sfreq=sfreq, bands=bands).mean(axis=0) for chunk in chunks],
        axis=0,
    )
    mean = segment_band.mean(axis=0)
    std = segment_band.std(axis=0)
    delta = segment_band[-1] - segment_band[0]
    return np.concatenate([mean, std, delta]).astype(np.float64, copy=False)


def _band_log_power_by_channel(
    x: np.ndarray,
    *,
    sfreq: float,
    bands: Sequence[tuple[str, float, float]],
    eps: float = 1e-12,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    freqs = np.fft.rfftfreq(arr.shape[1], d=1.0 / float(sfreq))
    spectrum = np.abs(np.fft.rfft(arr - arr.mean(axis=1, keepdims=True), axis=1)) ** 2
    out: list[np.ndarray] = []
    for _name, low, high in bands:
        mask = (freqs >= float(low)) & (freqs < float(high))
        if np.any(mask):
            out.append(np.log(spectrum[:, mask].mean(axis=1) + float(eps)))
        else:
            out.append(np.full(arr.shape[0], np.log(float(eps)), dtype=np.float64))
    return np.stack(out, axis=1)
