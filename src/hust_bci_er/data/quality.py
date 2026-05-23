"""Label-independent EEG window quality diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hust_bci_er.models.eeg_montage import HUST_30_A2_CHANNELS, indices_for_channel_names


FRONTAL_PROXY_CHANNELS = ("FP1", "FP2", "F7", "F8")


def window_quality_row(
    item: Mapping[str, Any],
    *,
    split: str,
    channel_names: Sequence[str] = HUST_30_A2_CHANNELS,
) -> dict[str, Any]:
    """Compute scale-aware, label-independent diagnostics for one EEG window."""
    x = np.asarray(item["x"], dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("EEG quality expects windows shaped [channels, time]")
    frontal_idx = list(indices_for_channel_names(channel_names, FRONTAL_PROXY_CHANNELS))
    ptp = np.ptp(x, axis=1)
    std = np.std(x, axis=1)
    power = float(np.mean(x * x))
    highfreq_proxy = float(np.mean(np.diff(x, axis=1) ** 2) / max(power, 1e-12)) if x.shape[1] > 1 else 0.0
    flat_fraction = float(np.mean(std < 1e-6))
    if x.shape[0] > 1 and x.shape[1] > 1:
        corr = np.corrcoef(x)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        offdiag = corr[~np.eye(corr.shape[0], dtype=bool)]
        mean_abs_corr = float(np.mean(np.abs(offdiag))) if offdiag.size else 0.0
    else:
        mean_abs_corr = 0.0

    frontal_ptp = float(np.max(ptp[frontal_idx])) if frontal_idx else 0.0
    global_ptp = float(np.max(ptp)) if ptp.size else 0.0
    # Higher is cleaner. This is a diagnostic score, not a model input.
    quality_score = 1.0 / (
        1.0
        + np.log1p(max(global_ptp, 0.0))
        + np.log1p(max(frontal_ptp, 0.0))
        + highfreq_proxy
        + flat_fraction
        + max(0.0, 0.2 - mean_abs_corr)
    )
    return {
        "split": split,
        "subject_id": item.get("subject_id", ""),
        "cohort": item.get("cohort", ""),
        "trial_id": item.get("trial_id", ""),
        "crop_id": item.get("crop_id", ""),
        "window_start_sec": item.get("window_start_sec", ""),
        "label": item.get("y", ""),
        "quality_score": float(quality_score),
        "frontal_ptp": frontal_ptp,
        "global_ptp": global_ptp,
        "highfreq_proxy": highfreq_proxy,
        "flat_channel_fraction": flat_fraction,
        "mean_abs_channel_corr": mean_abs_corr,
    }


def write_window_quality_csv(
    path: Path,
    split_windows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    fieldnames = [
        "split",
        "subject_id",
        "cohort",
        "trial_id",
        "crop_id",
        "window_start_sec",
        "label",
        "quality_score",
        "frontal_ptp",
        "global_ptp",
        "highfreq_proxy",
        "flat_channel_fraction",
        "mean_abs_channel_corr",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split, windows in split_windows.items():
            for item in windows:
                writer.writerow(window_quality_row(item, split=split))
