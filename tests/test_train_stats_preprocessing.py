from __future__ import annotations

import numpy as np

from hust_bci_er.data.quality import window_quality_row
from hust_bci_er.preprocessing.train_stats import (
    apply_channel_normalization,
    apply_robust_clip,
    fit_channel_normalization,
    fit_robust_clip,
)


def _window(x, *, subject="DEP1001", cohort="DEP", label=0, trial="t1", crop=0):
    return {
        "x": np.asarray(x, dtype=np.float32),
        "subject_id": subject,
        "cohort": cohort,
        "y": label,
        "trial_id": trial,
        "crop_id": crop,
        "window_start_sec": 0.0,
    }


def test_channel_normalization_uses_train_stats_on_other_split():
    train = [_window([[1, 2, 3], [10, 20, 30]])]
    val = [_window([[2, 3, 4], [20, 30, 40]], subject="HC1001", cohort="HC")]

    stats = fit_channel_normalization(train, method="median_MAD")
    apply_channel_normalization(val, stats)

    expected_ch0 = (np.asarray([2, 3, 4], dtype=np.float32) - 2.0) / 1.4826
    expected_ch1 = (np.asarray([20, 30, 40], dtype=np.float32) - 20.0) / 14.826
    assert np.allclose(val[0]["x"][0], expected_ch0, atol=1e-5)
    assert np.allclose(val[0]["x"][1], expected_ch1, atol=1e-5)


def test_robust_clip_reports_rates_without_dropping_windows():
    train = [_window([[0, 0, 100], [1, 1, 1]])]
    stats = fit_robust_clip(train, clip_n_mad=8)
    windows = [_window([[0, 0, 999], [1, 1, 1]])]

    clipped, audit = apply_robust_clip(windows, stats, split="train")

    assert len(clipped) == 1
    assert clipped[0]["x"][0, 2] <= stats["upper"][0]
    assert audit["n_windows"] == 1
    assert audit["clipped_window_fraction"] == 1.0
    assert audit["clip_window_rate_by_cohort"] == {"DEP": 1.0}


def test_window_quality_row_is_label_independent_shape_summary():
    row = window_quality_row(_window(np.eye(4, 8), label=1), split="train", channel_names=["FP1", "FP2", "F7", "F8"])

    assert row["split"] == "train"
    assert row["label"] == 1
    assert 0.0 < row["quality_score"] <= 1.0
    assert row["global_ptp"] >= row["frontal_ptp"]
