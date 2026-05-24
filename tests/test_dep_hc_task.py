from __future__ import annotations

import csv
import json
import zlib

import numpy as np
import pytest

from hust_bci_er.analysis.dep_hc_router import RouterSample, subject_threshold_diagnostic
from hust_bci_er.tasks.dep_hc.experiment import (
    evaluate_dep_hc_task,
    extract_dep_hc_task_features,
    fit_dep_hc_classifier,
    write_dep_hc_task_outputs,
)
from hust_bci_er.tasks.dep_hc.fusion import (
    _fusion_selection_key,
    _weight_candidates,
    evaluate_dep_hc_feature_fusion,
    evaluate_dep_hc_score_fusion,
)
from hust_bci_er.tasks.dep_hc.neural import (
    DEP_HC_NEURAL_MODELS,
    _apply_probability_calibrator,
    _build_model_kwargs,
    _fit_probability_calibrator,
    _output_logits,
    _subject_balanced_sample_weights,
    evaluate_dep_hc_neural_task,
)
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
from scripts.run_dep_hc_router import resolve_preprocessing
from scripts import run_dep_hc_neural_task, run_dep_hc_score_fusion_task


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


def _trial_crop_sample(subject: str, cohort: str, *, trial_id: str = "t0", crop_id: int = 0) -> RouterSample:
    scale = 0.5 if cohort == "HC" else 1.4
    return RouterSample(
        x=_window(f"{subject}-{cohort}-{trial_id}-{crop_id}", scale=scale),
        subject_id=subject,
        trial_id=f"{subject}_{trial_id}",
        crop_id=crop_id,
        window_start_sec=float(crop_id),
        cohort=cohort,
    )


def test_resolve_preprocessing_supports_explicit_raw_windows():
    assert resolve_preprocessing(None) == ["car", "zscore"]
    assert resolve_preprocessing(["none"]) == []
    assert resolve_preprocessing(["raw"]) == []
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_preprocessing(["none", "zscore"])


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
        _trial_crop_sample("HC101", "HC", crop_id=0),
        _trial_crop_sample("HC101", "HC", crop_id=1),
        _trial_crop_sample("DEP101", "DEP", crop_id=0),
        _trial_crop_sample("DEP101", "DEP", crop_id=1),
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
    assert result["metrics"]["classifier"] == "logistic"
    assert result["metrics"]["class_weight_mode"] == "uniform"
    assert result["metrics"]["n_eval_subjects"] == 2
    assert result["metrics"]["crop_combo_status"] == "computed"
    assert result["metrics"]["crop_combo_expected_trials"] == 1
    assert result["metrics"]["crop_combo_expected_crops"] == 2
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
    floor_summary = subject_threshold_diagnostic(samples, y_true, p_dep, objective="dep_recall_floor_0p8_hc")

    assert ba_summary["threshold"] == pytest.approx(0.5)
    assert ba_summary["balanced_accuracy"] == pytest.approx(0.75)
    assert min_recall_summary["threshold"] == pytest.approx(0.5)
    assert min_recall_summary["min_recall"] == pytest.approx(0.5)
    assert fixed_summary["threshold"] == pytest.approx(0.5)
    assert fixed_summary["objective"] == "fixed_0_5"
    assert floor_summary["objective"] == "dep_recall_floor_0p8_hc"
    assert floor_summary["dep_recall"] >= 0.8 * floor_summary["hc_recall"]
    with pytest.raises(ValueError, match="unknown threshold objective"):
        subject_threshold_diagnostic(samples, y_true, p_dep, objective="recall_gap")


def test_subject_threshold_diagnostic_respects_aggregation():
    samples = [
        _sample("HC001", "HC", crop_id=0),
        _sample("HC001", "HC", crop_id=1),
        _sample("HC001", "HC", crop_id=2),
        _sample("DEP001", "DEP", crop_id=0),
        _sample("DEP001", "DEP", crop_id=1),
        _sample("DEP001", "DEP", crop_id=2),
    ]
    y_true = np.array([0, 0, 0, 1, 1, 1])
    p_dep = np.array([0.49, 0.49, 0.99, 0.51, 0.51, 0.01])

    mean_summary = subject_threshold_diagnostic(samples, y_true, p_dep, objective="fixed_0_5", aggregation="mean")
    vote_summary = subject_threshold_diagnostic(samples, y_true, p_dep, objective="fixed_0_5", aggregation="vote_frac")

    assert mean_summary["balanced_accuracy"] == pytest.approx(0.0)
    assert vote_summary["balanced_accuracy"] == pytest.approx(1.0)
    assert vote_summary["aggregation"] == "vote_frac"
    with pytest.raises(ValueError, match="unknown subject aggregation"):
        subject_threshold_diagnostic(samples, y_true, p_dep, aggregation="filename")


def test_dep_hc_task_rejects_unknown_class_weight_mode():
    with pytest.raises(ValueError, match="unknown class_weight_mode"):
        evaluate_dep_hc_task(
            [_sample("HC001", "HC"), _sample("DEP001", "DEP")],
            [_sample("HC101", "HC"), _sample("DEP101", "DEP")],
            epochs=1,
            class_weight_mode="subject_id",
        )


def test_dep_hc_task_rejects_unknown_classifier():
    with pytest.raises(ValueError, match="unknown DEP/HC classifier"):
        fit_dep_hc_classifier(np.zeros((4, 2)), np.array([0, 0, 1, 1]), classifier="subject_id")


def test_dep_hc_task_sklearn_classifier_smoke():
    pytest.importorskip("sklearn")
    train = [
        _sample("HC001", "HC"),
        _sample("HC002", "HC"),
        _sample("DEP001", "DEP"),
        _sample("DEP002", "DEP"),
    ]
    eval_samples = [_sample("HC101", "HC"), _sample("DEP101", "DEP")]

    result = evaluate_dep_hc_task(
        train,
        eval_samples,
        feature_set="bandpower",
        sfreq=128.0,
        classifier="linear_svm",
        class_weight_mode="uniform",
    )

    assert result["metrics"]["classifier"] == "linear_svm"
    assert result["metrics"]["n_eval_subjects"] == 2


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
    assert result["metrics"]["fusion_weight_source"] == "validation_subjects"
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

    fixed_weight = evaluate_dep_hc_feature_fusion(
        train,
        eval_samples,
        val_samples=val,
        feature_sets=("bandpower", "time_frequency"),
        sfreq=128.0,
        epochs=5,
        weight_step=0.5,
        fixed_first_weight=0.25,
    )
    assert fixed_weight["metrics"]["fusion_weight_by_feature"] == {
        "bandpower": pytest.approx(0.25),
        "time_frequency": pytest.approx(0.75),
    }
    assert fixed_weight["metrics"]["fusion_weight_source"] == "fixed"
    assert fixed_weight["metrics"]["threshold_source"] == "validation_subjects"

    with pytest.raises(ValueError, match="fixed_first_weight"):
        evaluate_dep_hc_feature_fusion(
            train,
            eval_samples,
            val_samples=val,
            feature_sets=("bandpower", "time_frequency"),
            sfreq=128.0,
            epochs=5,
            fixed_first_weight=1.25,
        )


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
    floor_pass = {"balanced_accuracy": 0.7, "min_recall": 0.5, "recall_gap": 0.4, "hc_recall": 0.5, "dep_recall": 0.9}
    floor_fail = {"balanced_accuracy": 0.9, "min_recall": 0.4, "recall_gap": 0.5, "hc_recall": 1.0, "dep_recall": 0.4}
    assert _fusion_selection_key(
        floor_pass,
        threshold_objective="dep_recall_floor_0p8_hc",
        weight=0.5,
    ) > _fusion_selection_key(
        floor_fail,
        threshold_objective="dep_recall_floor_0p8_hc",
        weight=0.5,
    )


def test_dep_hc_score_fusion_reads_aligned_prediction_probabilities():
    rows_a = [
        {
            "subject_id": "HC101",
            "trial_id": "HC101_t0",
            "crop_id": "0",
            "window_start_sec": "0.0",
            "cohort": "HC",
            "y_true": "0",
            "p_dep": "0.20",
        },
        {
            "subject_id": "DEP101",
            "trial_id": "DEP101_t0",
            "crop_id": "0",
            "window_start_sec": "0.0",
            "cohort": "DEP",
            "y_true": "1",
            "p_dep": "0.70",
        },
    ]
    rows_b = [
        {**rows_a[0], "p_dep": "0.40"},
        {**rows_a[1], "p_dep": "0.90"},
    ]

    result = evaluate_dep_hc_score_fusion(
        {"deformer": rows_a, "traditional_tf": rows_b},
        weights={"deformer": 0.75, "traditional_tf": 0.25},
        threshold_objective="fixed_0_5",
        subject_aggregation="vote_frac",
    )

    assert result["metrics"]["score_fusion_components"] == ["deformer", "traditional_tf"]
    assert result["metrics"]["fusion_weight_source"] == "fixed"
    assert result["metrics"]["subject_ba"] == pytest.approx(1.0)
    assert result["metrics"]["crop_combo_status"] == "computed"
    assert result["metrics"]["crop_combo_expected_trials"] == 1
    assert result["metrics"]["crop_combo_expected_crops"] == 1
    by_subject = {row["subject_id"]: row["p_dep"] for row in result["prediction_rows"]}
    assert by_subject == {"HC101": "0.25", "DEP101": "0.75"}


def test_dep_hc_score_fusion_requires_validation_for_nonfixed_threshold():
    rows = [
        {
            "subject_id": "HC101",
            "trial_id": "HC101_t0",
            "crop_id": "0",
            "window_start_sec": "0.0",
            "cohort": "HC",
            "y_true": "0",
            "p_dep": "0.20",
        }
    ]
    with pytest.raises(ValueError, match="validation component predictions"):
        evaluate_dep_hc_score_fusion(
            {"a": rows, "b": rows},
            threshold_objective="balanced_accuracy",
        )


def test_dep_hc_score_fusion_validates_selection_component_names():
    rows = [
        {
            "subject_id": "HC101",
            "trial_id": "HC101_t0",
            "crop_id": "0",
            "window_start_sec": "0.0",
            "cohort": "HC",
            "y_true": "0",
            "p_dep": "0.20",
        }
    ]

    with pytest.raises(ValueError, match="component names must match"):
        evaluate_dep_hc_score_fusion(
            {"a": rows, "b": rows},
            val_component_prediction_rows={"a": rows, "typo": rows},
            threshold_objective="balanced_accuracy",
        )


def test_dep_hc_score_fusion_task_writes_jsonable_split_config(tmp_path):
    component_a = tmp_path / "component_a"
    component_b = tmp_path / "component_b"
    _write_score_component_run(component_a, hc_p_dep="0.20", dep_p_dep="0.70")
    _write_score_component_run(component_b, hc_p_dep="0.30", dep_p_dep="0.80")
    out_dir = tmp_path / "fusion"

    assert (
        run_dep_hc_score_fusion_task.main(
            [
                "--out-dir",
                str(out_dir),
                "--component-run-dir",
                f"deformer={component_a}",
                "--component-run-dir",
                f"traditional={component_b}",
                "--threshold-objective",
                "fixed_0_5",
            ]
        )
        == 0
    )

    payload = json.loads((out_dir / "dep_hc_task_diagnostic.json").read_text(encoding="utf-8"))
    config = payload["config"]
    assert config["protocol"] == "p2"
    assert config["split_id"] == "dep_hc_router_p2_holdout123_train42"
    assert config["holdout_seed"] == 123
    assert config["component_roots"] == {
        "deformer": str(component_a),
        "traditional": str(component_b),
    }
    assert config["selection_component_roots"] == {}


def test_dep_hc_score_fusion_task_inherits_p3_split_metadata(tmp_path):
    component_a = tmp_path / "component_a"
    component_b = tmp_path / "component_b"
    _write_score_component_run(
        component_a,
        hc_p_dep="0.20",
        dep_p_dep="0.70",
        config_overrides={
            "protocol": "p3",
            "split_id": "dep_hc_router_p3_outer1_inner2_seed42_123",
            "outer_fold": 1,
            "inner_fold": 2,
            "outer_folds": 5,
            "inner_folds": 3,
            "outer_seed": 42,
            "inner_seed": 123,
            "eval_scope": "p3_inner_validation",
        },
    )
    _write_score_component_run(
        component_b,
        hc_p_dep="0.30",
        dep_p_dep="0.80",
        config_overrides={
            "protocol": "p3",
            "split_id": "dep_hc_router_p3_outer1_inner2_seed42_123",
            "outer_fold": 1,
            "inner_fold": 2,
            "outer_folds": 5,
            "inner_folds": 3,
            "outer_seed": 42,
            "inner_seed": 123,
            "eval_scope": "p3_inner_validation",
        },
    )
    out_dir = tmp_path / "fusion"

    assert (
        run_dep_hc_score_fusion_task.main(
            [
                "--out-dir",
                str(out_dir),
                "--component-run-dir",
                f"deformer={component_a}",
                "--component-run-dir",
                f"traditional={component_b}",
                "--threshold-objective",
                "fixed_0_5",
            ]
        )
        == 0
    )

    config = json.loads((out_dir / "dep_hc_task_diagnostic.json").read_text(encoding="utf-8"))["config"]
    assert config["protocol"] == "p3"
    assert config["outer_fold"] == 1
    assert config["inner_fold"] == 2
    assert config["eval_scope"] == "p3_inner_validation"


def test_dep_hc_score_fusion_task_allows_p3_component_training_seed_mismatch(tmp_path):
    component_a = tmp_path / "component_a"
    component_b = tmp_path / "component_b"
    p3_split = {
        "protocol": "p3",
        "split_id": "dep_hc_router_p3_outer2_final_seed42",
        "seed": 10121,
        "fold": 0,
        "holdout_seed": 999,
        "outer_fold": 2,
        "inner_fold": None,
        "outer_folds": 5,
        "inner_folds": 3,
        "outer_seed": 42,
        "inner_seed": 123,
        "eval_scope": "p3_outer_test",
    }
    _write_score_component_run(
        component_a,
        hc_p_dep="0.20",
        dep_p_dep="0.70",
        config_overrides=p3_split,
    )
    _write_score_component_run(
        component_b,
        hc_p_dep="0.30",
        dep_p_dep="0.80",
        config_overrides={**p3_split, "seed": 11221},
    )
    out_dir = tmp_path / "fusion"

    assert (
        run_dep_hc_score_fusion_task.main(
            [
                "--out-dir",
                str(out_dir),
                "--component-run-dir",
                f"deformer={component_a}",
                "--component-run-dir",
                f"tsception={component_b}",
                "--threshold-objective",
                "fixed_0_5",
            ]
        )
        == 0
    )

    config = json.loads((out_dir / "dep_hc_task_diagnostic.json").read_text(encoding="utf-8"))["config"]
    assert config["split_id"] == "dep_hc_router_p3_outer2_final_seed42"
    assert config["outer_fold"] == 2
    assert "seed" not in config


def _write_score_component_run(root, *, hc_p_dep: str, dep_p_dep: str, config_overrides: dict | None = None) -> None:
    root.mkdir()
    rows = [
        {
            "subject_id": "HC101",
            "trial_id": "HC101_t0",
            "crop_id": "0",
            "window_start_sec": "0.0",
            "cohort": "HC",
            "y_true": "0",
            "p_dep": hc_p_dep,
        },
        {
            "subject_id": "DEP101",
            "trial_id": "DEP101_t0",
            "crop_id": "0",
            "window_start_sec": "0.0",
            "cohort": "DEP",
            "y_true": "1",
            "p_dep": dep_p_dep,
        },
    ]
    with (root / "dep_hc_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "dep_hc_task_diagnostic.json").write_text(
        json.dumps(
            {
                "task": "dep_hc",
                "config": {
                    "protocol": "p2",
                    "split_id": "dep_hc_router_p2_holdout123_train42",
                    "seed": 42,
                    "n_holdout_subjects": 12,
                    "holdout_seed": 123,
                    **(config_overrides or {}),
                },
                "metrics": {},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_dep_hc_neural_task_smoke_uses_cohort_target():
    pytest.importorskip("torch")
    train = [
        _sample("HC001", "HC"),
        _sample("HC002", "HC"),
        _sample("DEP001", "DEP"),
        _sample("DEP002", "DEP"),
    ]
    val = [_sample("HC011", "HC"), _sample("DEP011", "DEP")]
    eval_samples = [_sample("HC101", "HC"), _sample("DEP101", "DEP")]

    result = evaluate_dep_hc_neural_task(
        train,
        eval_samples,
        val_samples=val,
        model_name="eegnet",
        epochs=1,
        batch_size=2,
        seed=7,
    )

    assert result["metrics"]["task"] == "dep_hc"
    assert result["metrics"]["model_name"] == "eegnet"
    assert result["metrics"]["n_eval_subjects"] == 2
    assert {"HC101", "DEP101"} == {row["subject_id"] for row in result["subject_rows"]}


def test_dep_hc_neural_task_uses_actual_crop_combo_shape():
    pytest.importorskip("torch")
    train = [
        _sample("HC001", "HC"),
        _sample("HC002", "HC"),
        _sample("DEP001", "DEP"),
        _sample("DEP002", "DEP"),
    ]
    val = [_sample("HC011", "HC"), _sample("DEP011", "DEP")]
    eval_samples = [
        _trial_crop_sample(subject, cohort, crop_id=crop_id)
        for subject, cohort in [("HC101", "HC"), ("DEP101", "DEP")]
        for crop_id in range(6)
    ]

    result = evaluate_dep_hc_neural_task(
        train,
        eval_samples,
        val_samples=val,
        model_name="eegnet",
        epochs=1,
        batch_size=2,
        seed=7,
    )

    assert result["metrics"]["crop_combo_status"] == "computed"
    assert result["metrics"]["crop_combo_expected_trials"] == 1
    assert result["metrics"]["crop_combo_expected_crops"] == 6


def test_run_dep_hc_neural_task_rejects_nonpositive_stride(tmp_path):
    with pytest.raises(SystemExit):
        run_dep_hc_neural_task.main(["--out-dir", str(tmp_path), "--stride-sec", "0"])


def test_dep_hc_neural_task_supports_calibration_and_subject_balanced_sampling():
    pytest.importorskip("torch")
    train = [
        _sample("HC001", "HC", crop_id=0),
        _sample("HC001", "HC", crop_id=1),
        _sample("HC002", "HC", crop_id=0),
        _sample("DEP001", "DEP", crop_id=0),
    ]
    val = [_sample("HC011", "HC"), _sample("DEP011", "DEP")]
    eval_samples = [_sample("HC101", "HC"), _sample("DEP101", "DEP")]

    result = evaluate_dep_hc_neural_task(
        train,
        eval_samples,
        val_samples=val,
        model_name="eegnet",
        epochs=1,
        batch_size=2,
        seed=7,
        calibration_method="temperature",
        sampling_strategy="subject_balanced",
    )

    assert result["metrics"]["calibration_method"] == "temperature"
    assert result["metrics"]["sampling_strategy"] == "subject_balanced"
    assert "temperature" in result["metrics"]["calibration_params"]
    assert "window_brier_raw" in result["metrics"]


def test_dep_hc_neural_calibrators_and_subject_weights_are_metadata_safe():
    torch = pytest.importorskip("torch")
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.2, 0.8, 0.6, 0.9])

    temp = _fit_probability_calibrator(labels, probs, method="temperature")
    platt = _fit_probability_calibrator(labels, probs, method="platt")

    assert temp["method"] == "temperature"
    assert platt["method"] == "platt"
    assert _apply_probability_calibrator(probs, temp).shape == probs.shape
    assert _apply_probability_calibrator(probs, platt).shape == probs.shape

    samples = [
        _sample("HC001", "HC", crop_id=0),
        _sample("HC001", "HC", crop_id=1),
        _sample("HC002", "HC", crop_id=0),
        _sample("DEP001", "DEP", crop_id=0),
        _sample("DEP001", "DEP", crop_id=1),
    ]
    weights = _subject_balanced_sample_weights(samples, torch.tensor([0, 0, 0, 1, 1]))
    by_subject: dict[str, float] = {}
    for sample, weight in zip(samples, weights.tolist()):
        by_subject[sample.subject_id] = by_subject.get(sample.subject_id, 0.0) + float(weight)

    assert by_subject["HC001"] == pytest.approx(by_subject["HC002"])
    assert by_subject["DEP001"] == pytest.approx(0.5)


def test_dep_hc_neural_task_supports_factory_backbone_dict_logits():
    torch = pytest.importorskip("torch")
    assert {
        "cbramod",
        "conformer_lite",
        "dgcnn",
        "fbcnet",
        "fbstcnet",
        "riemannian_tangent",
        "srf_fbstcnet_gate",
        "srfnet",
        "tri_context_gate",
    } <= DEP_HC_NEURAL_MODELS

    logits = torch.randn(2, 2)
    assert torch.equal(_output_logits(logits), logits)
    assert torch.equal(_output_logits({"logits": logits, "aux": torch.ones(2)}), logits)
    with pytest.raises(KeyError, match="logits"):
        _output_logits({"score": torch.ones(2)})

    dgcnn_kwargs = _build_model_kwargs("dgcnn")
    assert dgcnn_kwargs["channel_montage"] == "hust_30_a2"
    fbstcnet_kwargs = _build_model_kwargs("fbstcnet", {"variant": "C", "gamma": 123})
    assert fbstcnet_kwargs["variant"] == "C"
    assert fbstcnet_kwargs["gamma"] == 123
    cbramod_kwargs = _build_model_kwargs("cbramod")
    assert cbramod_kwargs["classifier_pooling"] == "mean"
    assert cbramod_kwargs["d_model"] == 64
    route_cbramod_kwargs = _build_model_kwargs("cbramod", {"patch_size": 250, "d_model": 250, "nhead": 10})
    assert route_cbramod_kwargs["conv_out_channels"] == 25
    assert route_cbramod_kwargs["group_norm_groups"] == 5
    from hust_bci_er.models.factory import build_model

    build_model("cbramod", n_channels=30, n_times=2500, n_classes=2, **route_cbramod_kwargs)
    fbcnet_kwargs = _build_model_kwargs("fbcnet")
    assert fbcnet_kwargs["n_segments"] == 5
    gate_kwargs = _build_model_kwargs("tri_context_gate")
    assert gate_kwargs["gate_hidden_dim"] == 16
    assert gate_kwargs["fbstcnet"]["variant"] == "M"


def test_dep_hc_sample_maker_supports_sliding_windows():
    from scripts.run_dep_hc_router import _make_samples

    trial = {
        "x": _window("sliding-source", scale=1.0).repeat(10, axis=1)[:, :2500],
        "y": 0,
        "subject_id": "HC001",
        "cohort": "HC",
        "trial_id": "HC001_neu1",
    }

    samples = _make_samples(
        [trial],
        preprocessing=[],
        source_trial_sec=10,
        window_sec=6,
        n_crops=5,
        stride_sec=1,
        ea_transform=None,
    )

    assert len(samples) == 5
    assert [sample.crop_id for sample in samples] == [0, 1, 2, 3, 4]
    assert [sample.window_start_sec for sample in samples] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert {sample.x.shape for sample in samples} == {(30, 1500)}


def test_dep_hc_neural_task_rejects_unknown_model_name():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="unknown DEP/HC neural model_name"):
        evaluate_dep_hc_neural_task(
            [_sample("HC001", "HC"), _sample("DEP001", "DEP")],
            [_sample("HC101", "HC"), _sample("DEP101", "DEP")],
            val_samples=[_sample("HC011", "HC"), _sample("DEP011", "DEP")],
            model_name="subject_id",
            epochs=1,
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
