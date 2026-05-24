from __future__ import annotations

import csv
import json
import zlib

import numpy as np
import pytest

from hust_bci_er.analysis.dep_hc_router import (
    RouterSample,
    _threshold_boundary_candidates,
    aggregate_subject_rows,
    balanced_accuracy_binary,
    bandpower_features,
    covariance_tangent_features,
    crop_combo_subject_metrics,
    evaluate_router,
    extract_router_features,
    router_features_for_sample,
    select_subject_threshold,
    write_router_outputs,
)
from scripts.run_dep_hc_router import resolve_preprocessing
from scripts import run_dep_hc_router


def _sample(subject: str, cohort: str, *, value: float, crop_id: int = 0) -> RouterSample:
    seed = zlib.crc32(f"{subject}|{cohort}|{value}|{crop_id}".encode("utf-8"))
    rng = np.random.default_rng(seed)
    scale = 0.20 if cohort == "HC" else 1.20
    base = rng.normal(loc=0.0, scale=scale, size=(4, 64)).astype(np.float32)
    base += np.float32(value * 0.01)
    return RouterSample(
        x=base,
        subject_id=subject,
        trial_id=f"{subject}_t{crop_id}",
        crop_id=crop_id,
        window_start_sec=float(crop_id),
        cohort=cohort,
    )


def test_router_features_are_numeric_and_metadata_free():
    sample = _sample("DEP001", "DEP", value=1.0)

    cov = covariance_tangent_features(sample.x)
    bands = bandpower_features(sample.x, sfreq=64.0)
    features = extract_router_features([sample])

    assert cov.shape == (10,)
    assert bands.shape == (20,)
    assert features.shape == (1, 30)
    assert router_features_for_sample(sample.x, feature_set="cov_tangent").shape == (10,)
    assert router_features_for_sample(sample.x, feature_set="bandpower").shape == (20,)
    assert np.isfinite(features).all()


def test_evaluate_router_reports_subject_level_metrics():
    train = [
        _sample("HC001", "HC", value=-1.0, crop_id=0),
        _sample("HC002", "HC", value=-0.9, crop_id=0),
        _sample("DEP001", "DEP", value=1.0, crop_id=0),
        _sample("DEP002", "DEP", value=0.9, crop_id=0),
    ]
    eval_samples = [
        _sample("HC101", "HC", value=-1.1, crop_id=0),
        _sample("HC101", "HC", value=-1.0, crop_id=1),
        _sample("DEP101", "DEP", value=1.1, crop_id=0),
        _sample("DEP101", "DEP", value=1.0, crop_id=1),
    ]

    result = evaluate_router(train, eval_samples, epochs=80, lr=0.1, l2=1e-4, feature_set="cov_tangent_bandpower")

    assert result["metrics"]["n_eval_subjects"] == 2
    assert result["metrics"]["feature_set"] == "cov_tangent_bandpower"
    assert result["metrics"]["threshold_source"] == "fixed_0.5"
    assert result["metrics"]["subject_ba"] >= 0.5
    assert {row["subject_id"] for row in result["subject_rows"]} == {"DEP101", "HC101"}
    assert {"p_dep", "p_hc", "confidence", "predicted_cohort"}.issubset(result["prediction_rows"][0])


def test_aggregate_subject_rows_uses_mean_probability():
    rows = [
        {"subject_id": "DEP001", "cohort": "DEP", "y_true": "1", "p_dep": "0.8"},
        {"subject_id": "DEP001", "cohort": "DEP", "y_true": "1", "p_dep": "0.6"},
        {"subject_id": "HC001", "cohort": "HC", "y_true": "0", "p_dep": "0.2"},
    ]

    subjects = aggregate_subject_rows(rows)

    by_id = {row["subject_id"]: row for row in subjects}
    assert by_id["DEP001"]["predicted_cohort"] == "DEP"
    assert by_id["DEP001"]["threshold"] == "0.5"
    assert by_id["DEP001"]["n_windows"] == "2"
    assert by_id["HC001"]["predicted_cohort"] == "HC"


def test_aggregate_subject_rows_uses_selected_threshold():
    rows = [
        {"subject_id": "DEP001", "cohort": "DEP", "y_true": "1", "p_dep": "0.8"},
        {"subject_id": "DEP001", "cohort": "DEP", "y_true": "1", "p_dep": "0.6"},
    ]

    subjects = aggregate_subject_rows(rows, threshold=0.75)

    assert subjects[0]["predicted_cohort"] == "HC"
    assert subjects[0]["correct"] == "0"
    assert subjects[0]["threshold"] == "0.75"


def test_aggregate_subject_rows_supports_alternate_rules():
    rows = [
        {"subject_id": "DEP001", "cohort": "DEP", "y_true": "1", "p_dep": "0.95"},
        {"subject_id": "DEP001", "cohort": "DEP", "y_true": "1", "p_dep": "0.10"},
        {"subject_id": "DEP001", "cohort": "DEP", "y_true": "1", "p_dep": "0.20"},
        {"subject_id": "DEP001", "cohort": "DEP", "y_true": "1", "p_dep": "0.90"},
    ]

    median = aggregate_subject_rows(rows, aggregation="median")
    vote = aggregate_subject_rows(rows, aggregation="vote_frac")
    trimmed = aggregate_subject_rows(rows, aggregation="trimmed_mean")

    assert median[0]["subject_score_p_dep"] == "0.55"
    assert median[0]["predicted_cohort"] == "DEP"
    assert vote[0]["subject_score_p_dep"] == "0.5"
    assert vote[0]["predicted_cohort"] == "DEP"
    assert trimmed[0]["subject_aggregation"] == "trimmed_mean"
    with pytest.raises(ValueError, match="unknown subject aggregation"):
        aggregate_subject_rows(rows, aggregation="subject_id")


def test_crop_combo_subject_metrics_enumerates_heldout_10s_assignments():
    rows = [
        {"subject_id": "HC001", "trial_id": "t0", "crop_id": "0", "cohort": "HC", "y_true": "0", "p_dep": "0.1"},
        {"subject_id": "HC001", "trial_id": "t0", "crop_id": "1", "cohort": "HC", "y_true": "0", "p_dep": "0.9"},
        {"subject_id": "HC001", "trial_id": "t1", "crop_id": "0", "cohort": "HC", "y_true": "0", "p_dep": "0.1"},
        {"subject_id": "HC001", "trial_id": "t1", "crop_id": "1", "cohort": "HC", "y_true": "0", "p_dep": "0.1"},
        {"subject_id": "DEP001", "trial_id": "t0", "crop_id": "0", "cohort": "DEP", "y_true": "1", "p_dep": "0.9"},
        {"subject_id": "DEP001", "trial_id": "t0", "crop_id": "1", "cohort": "DEP", "y_true": "1", "p_dep": "0.9"},
        {"subject_id": "DEP001", "trial_id": "t1", "crop_id": "0", "cohort": "DEP", "y_true": "1", "p_dep": "0.1"},
        {"subject_id": "DEP001", "trial_id": "t1", "crop_id": "1", "cohort": "DEP", "y_true": "1", "p_dep": "0.9"},
    ]

    metrics = crop_combo_subject_metrics(rows, threshold=0.5, aggregation="mean", n_trials=2, n_crops=2)

    assert metrics["crop_combo_status"] == "computed"
    assert metrics["crop_combo_n_assignments"] == 4
    assert metrics["crop_combo_complete_subjects"] == 2
    assert metrics["crop_combo_expected_ba"] == pytest.approx(0.75)
    assert metrics["crop_combo_worst_ba"] == pytest.approx(0.5)
    assert metrics["crop_combo_best_ba"] == pytest.approx(1.0)


def test_vote_frac_threshold_candidates_include_discrete_boundaries():
    candidates = _threshold_boundary_candidates(
        {
            "HC001": [0.1, 0.8, 0.9],
            "DEP001": [0.2, 0.3, 0.7],
        },
        aggregation="vote_frac",
    )

    assert 1.0 / 3.0 in candidates
    assert 2.0 / 3.0 in candidates
    assert 0.8 in candidates
    assert _threshold_boundary_candidates({"HC001": [0.1, 0.8, 0.9]}, aggregation="mean") == []


def test_resolve_preprocessing_overrides_default_when_cli_supplies_values():
    assert resolve_preprocessing(None) == ["car", "zscore"]
    assert resolve_preprocessing(["bandpass"]) == ["bandpass"]
    assert resolve_preprocessing(["bandpass", "zscore"]) == ["bandpass", "zscore"]


def test_run_dep_hc_router_rejects_unknown_preprocessing(tmp_path):
    with pytest.raises(ValueError, match="unknown preprocessing step"):
        run_dep_hc_router.main(["--out-dir", str(tmp_path), "--preprocessing", "zscroe"])


def test_select_subject_threshold_uses_validation_subjects():
    samples = [
        _sample("HC001", "HC", value=-1.0, crop_id=0),
        _sample("HC001", "HC", value=-1.0, crop_id=1),
        _sample("DEP001", "DEP", value=1.0, crop_id=0),
        _sample("DEP001", "DEP", value=1.0, crop_id=1),
    ]
    y_true = np.array([0, 0, 1, 1])
    p_dep = np.array([0.40, 0.42, 0.60, 0.62])

    threshold = select_subject_threshold(samples, y_true, p_dep)

    assert 0.41 <= threshold <= 0.61


def test_balanced_accuracy_binary_handles_two_cohorts():
    assert balanced_accuracy_binary(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])) == 0.75


def test_write_router_outputs(tmp_path):
    result = {
        "metrics": {"subject_ba": 1.0, "window_ba": 1.0},
        "prediction_rows": [
            {
                "subject_id": "DEP001",
                "trial_id": "DEP001_t0",
                "crop_id": "0",
                "window_start_sec": "0.0",
                "cohort": "DEP",
                "y_true": "1",
                "p_hc": "0.1",
                "p_dep": "0.9",
                "confidence": "0.9",
                "y_pred": "1",
                "predicted_cohort": "DEP",
            }
        ],
        "subject_rows": [
            {
                "subject_id": "DEP001",
                "cohort": "DEP",
                "y_true": "1",
                "mean_p_dep": "0.9",
                "mean_p_hc": "0.1",
                "confidence": "0.9",
                "predicted_cohort": "DEP",
                "correct": "1",
                "n_windows": "1",
            }
        ],
    }

    write_router_outputs(result, tmp_path, config={"protocol": "p1"})

    assert (tmp_path / "router_diagnostic.md").exists()
    payload = json.loads((tmp_path / "router_diagnostic.json").read_text(encoding="utf-8"))
    with (tmp_path / "router_predictions.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert payload["task"] == "dep_hc_router"
    assert rows[0]["p_dep"] == "0.9"
