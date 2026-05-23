from __future__ import annotations

import zlib
import json

import numpy as np
import pytest

from hust_bci_er.analysis.dep_hc_router import RouterSample, subject_threshold_diagnostic
from hust_bci_er.tasks.dep_hc.experiment import evaluate_dep_hc_task, extract_dep_hc_task_features, write_dep_hc_task_outputs
from hust_bci_er.tasks.dep_hc.fusion import _fusion_selection_key, _weight_candidates, evaluate_dep_hc_feature_fusion
from hust_bci_er.tasks.dep_hc.channel_graph import (
    channel_graph_adjacency,
    channel_graph_metadata,
    graph_edge_indices,
    regional_connectivity_features,
)
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
    traditional_graph = dep_hc_features(x, feature_set="traditional_graph", sfreq=128.0)
    connectivity = connectivity_summary_features(x)
    regional = regional_connectivity_features(x)
    asymmetry = asymmetry_features(x, sfreq=128.0)
    time_frequency = time_frequency_summary_features(x, sfreq=128.0)

    assert traditional.ndim == 1
    assert connectivity.shape == (30 * 29,)
    assert regional.shape == (30,)
    assert asymmetry.shape == (13,)
    assert time_frequency.shape == (15,)
    assert traditional_graph.shape[0] == traditional.shape[0] + regional.shape[0]
    assert np.isfinite(traditional).all()


def test_traditional_asymmetry_requires_hust_montage():
    with pytest.raises(ValueError, match="hust_30_a2"):
        dep_hc_features(_window("bad-montage"), feature_set="traditional", channel_montage="sequential")


def test_graph_connectivity_requires_hust_montage():
    with pytest.raises(ValueError, match="HUST 30-channel A2"):
        dep_hc_features(_window("bad-graph-montage"), feature_set="graph_connectivity", channel_montage="sequential")
    with pytest.raises(ValueError, match="HUST 30-channel A2"):
        dep_hc_features(_window("empty-graph-montage"), feature_set="graph_connectivity", channel_montage="")


def test_hust_channel_graph_uses_pdf_order_and_region_priors():
    table = channel_graph_metadata()["channels"]
    adjacency = channel_graph_adjacency()
    edges = graph_edge_indices(include_self=False)

    assert table[0]["name"] == "FP1"
    assert table[-1]["name"] == "O2"
    assert adjacency.shape == (30, 30)
    assert np.allclose(adjacency, adjacency.T)
    assert len(edges) > 30
    assert adjacency[0, 1] == 1.0
    assert adjacency[3, 5] == 1.0


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

    features = extract_dep_hc_task_features(train, feature_set="traditional_graph", sfreq=128.0)
    result = evaluate_dep_hc_task(
        train,
        eval_samples,
        feature_set="traditional_graph",
        sfreq=128.0,
        epochs=40,
        lr=0.1,
        l2=1e-3,
        class_weight_mode="uniform",
    )

    assert features.shape[0] == 4
    assert result["metrics"]["task"] == "dep_hc"
    assert result["metrics"]["feature_set"] == "traditional_graph"
    assert result["metrics"]["class_weight_mode"] == "uniform"
    assert result["metrics"]["n_eval_subjects"] == 2
    assert {"DEP101", "HC101"} == {row["subject_id"] for row in result["subject_rows"]}


def test_subject_threshold_objective_is_validation_only_summary():
    samples = [
        _sample("HC001", "HC"),
        _sample("HC002", "HC"),
        _sample("DEP001", "DEP"),
        _sample("DEP002", "DEP"),
    ]
    y_true = np.array([0, 0, 1, 1])
    p_dep = np.array([0.40, 0.62, 0.58, 0.70])

    ba_summary = subject_threshold_diagnostic(samples, y_true, p_dep, objective="balanced_accuracy")
    min_recall_summary = subject_threshold_diagnostic(samples, y_true, p_dep, objective="min_recall")
    fixed_summary = subject_threshold_diagnostic(samples, y_true, p_dep, objective="fixed_0_5")

    assert ba_summary["threshold"] == pytest.approx(0.5)
    assert ba_summary["balanced_accuracy"] == pytest.approx(0.75)
    assert min_recall_summary["threshold"] == pytest.approx(0.5)
    assert min_recall_summary["min_recall"] == pytest.approx(0.5)
    assert fixed_summary["threshold"] == pytest.approx(0.5)
    assert fixed_summary["objective"] == "fixed_0_5"
    with pytest.raises(ValueError, match="unknown threshold objective"):
        subject_threshold_diagnostic(samples, y_true, p_dep, objective="recall_gap")


def test_dep_hc_task_rejects_unknown_class_weight_mode():
    with pytest.raises(ValueError, match="unknown class_weight_mode"):
        evaluate_dep_hc_task(
            [_sample("HC001", "HC"), _sample("DEP001", "DEP")],
            [_sample("HC101", "HC"), _sample("DEP101", "DEP")],
            epochs=1,
            class_weight_mode="subject_id",
        )


def test_dep_hc_feature_fusion_selects_validation_weight():
    train = [
        _sample("HC001", "HC"),
        _sample("HC002", "HC"),
        _sample("DEP001", "DEP"),
        _sample("DEP002", "DEP"),
    ]
    val = [
        _sample("HC011", "HC", crop_id=0),
        _sample("HC011", "HC", crop_id=1),
        _sample("DEP011", "DEP", crop_id=0),
        _sample("DEP011", "DEP", crop_id=1),
    ]
    eval_samples = [
        _sample("HC101", "HC", crop_id=0),
        _sample("DEP101", "DEP", crop_id=0),
    ]

    result = evaluate_dep_hc_feature_fusion(
        train,
        eval_samples,
        val_samples=val,
        feature_sets=("bandpower", "time_frequency"),
        sfreq=128.0,
        epochs=5,
        weight_step=0.5,
    )

    assert result["metrics"]["task"] == "dep_hc"
    assert result["metrics"]["fusion_feature_sets"] == ["bandpower", "time_frequency"]
    assert set(result["metrics"]["fusion_weight_by_feature"]) == {"bandpower", "time_frequency"}
    assert result["metrics"]["threshold_source"] == "validation_subjects"

    fixed = evaluate_dep_hc_feature_fusion(
        train,
        eval_samples,
        val_samples=val,
        feature_sets=("bandpower", "time_frequency"),
        sfreq=128.0,
        epochs=5,
        weight_step=0.5,
        threshold_objective="fixed_0_5",
    )
    assert fixed["metrics"]["threshold"] == pytest.approx(0.5)
    assert fixed["metrics"]["threshold_source"] == "fixed_0.5"


def test_dep_hc_fusion_weight_selection_respects_objective_and_endpoints():
    assert _weight_candidates(0.4) == (0.0, 0.4, 0.8, 1.0)
    high_ba = {"balanced_accuracy": 0.9, "min_recall": 0.4, "recall_gap": 0.8}
    high_min = {"balanced_accuracy": 0.7, "min_recall": 0.7, "recall_gap": 0.0}

    assert _fusion_selection_key(high_ba, threshold_objective="balanced_accuracy", weight=0.5) > _fusion_selection_key(
        high_min,
        threshold_objective="balanced_accuracy",
        weight=0.5,
    )
    assert _fusion_selection_key(high_ba, threshold_objective="fixed_0_5", weight=0.5) > _fusion_selection_key(
        high_min,
        threshold_objective="fixed_0_5",
        weight=0.5,
    )
    assert _fusion_selection_key(high_min, threshold_objective="min_recall", weight=0.5) > _fusion_selection_key(
        high_ba,
        threshold_objective="min_recall",
        weight=0.5,
    )


def test_write_dep_hc_task_outputs_uses_task_specific_payload(tmp_path):
    result = {
        "metrics": {"task": "dep_hc", "subject_ba": 1.0},
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
                "threshold": "0.5",
                "confidence": "0.9",
                "predicted_cohort": "DEP",
                "correct": "1",
                "n_windows": "1",
            }
        ],
    }

    write_dep_hc_task_outputs(result, tmp_path, config={"feature_set": "traditional"})

    payload = json.loads((tmp_path / "dep_hc_task_diagnostic.json").read_text(encoding="utf-8"))
    assert payload["task"] == "dep_hc"
    assert (tmp_path / "dep_hc_predictions.csv").exists()
    assert (tmp_path / "dep_hc_subject_metrics.csv").exists()
