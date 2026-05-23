from __future__ import annotations

import zlib

import numpy as np

from hust_bci_er.analysis.dep_hc_router import RouterSample
from hust_bci_er.tasks.dep_hc.experiment import evaluate_dep_hc_task, extract_dep_hc_task_features
from hust_bci_er.tasks.dep_hc.features import (
    asymmetry_features,
    connectivity_summary_features,
    dep_hc_features,
    time_frequency_summary_features,
)


def _window(seed_text: str, *, scale: float = 1.0) -> np.ndarray:
    seed = zlib.crc32(seed_text.encode("utf-8"))
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=(30, 256)).astype(np.float32)


def _sample(subject: str, cohort: str, *, crop_id: int = 0) -> RouterSample:
    scale = 0.5 if cohort == "HC" else 1.4
    return RouterSample(
        x=_window(f"{subject}-{cohort}-{crop_id}", scale=scale),
        subject_id=subject,
        trial_id=f"{subject}_t{crop_id}",
        crop_id=crop_id,
        window_start_sec=float(crop_id),
        cohort=cohort,
    )


def test_dep_hc_task_feature_sets_are_numeric_and_metadata_free():
    x = _window("feature-smoke")

    traditional = dep_hc_features(x, feature_set="traditional", sfreq=128.0)
    connectivity = connectivity_summary_features(x)
    asymmetry = asymmetry_features(x, sfreq=128.0)
    time_frequency = time_frequency_summary_features(x, sfreq=128.0)

    assert traditional.ndim == 1
    assert connectivity.shape == (30 * 29,)
    assert asymmetry.shape == (13,)
    assert time_frequency.shape == (15,)
    assert np.isfinite(traditional).all()


def test_dep_hc_task_feature_matrix_and_eval():
    train = [
        _sample("HC001", "HC"),
        _sample("HC002", "HC"),
        _sample("DEP001", "DEP"),
        _sample("DEP002", "DEP"),
    ]
    eval_samples = [
        _sample("HC101", "HC", crop_id=0),
        _sample("HC101", "HC", crop_id=1),
        _sample("DEP101", "DEP", crop_id=0),
        _sample("DEP101", "DEP", crop_id=1),
    ]

    features = extract_dep_hc_task_features(train, feature_set="traditional_time_frequency", sfreq=128.0)
    result = evaluate_dep_hc_task(
        train,
        eval_samples,
        feature_set="traditional_time_frequency",
        sfreq=128.0,
        epochs=40,
        lr=0.1,
        l2=1e-3,
    )

    assert features.shape[0] == 4
    assert result["metrics"]["task"] == "dep_hc"
    assert result["metrics"]["feature_set"] == "traditional_time_frequency"
    assert result["metrics"]["n_eval_subjects"] == 2
    assert {"DEP101", "HC101"} == {row["subject_id"] for row in result["subject_rows"]}

