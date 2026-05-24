from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import summarize_dep_hc_diagnostics


def _write_run(
    root: Path,
    *,
    feature_set: str | None = "traditional",
    fusion: bool = False,
    fusion_features: list[str] | None = None,
    subject_ba: float = 0.75,
    hc_recall: float = 0.5,
    dep_recall: float = 1.0,
) -> None:
    root.mkdir(parents=True)
    metrics = {
        "task": "dep_hc",
        "subject_ba": subject_ba,
        "hc_subject_recall": hc_recall,
        "dep_subject_recall": dep_recall,
        "n_eval_subjects": 2,
        "threshold_objective": "fixed_0_5",
        "threshold_source": "fixed_0.5",
        "subject_aggregation": "mean",
    }
    config = {"protocol": "p2", "split_id": "split_h123", "feature_set": feature_set}
    if fusion:
        features = fusion_features or ["traditional", "time_frequency"]
        metrics.update(
            {
                "fusion_feature_sets": features,
                "fusion_weight_by_feature": {feature: 0.5 for feature in features},
                "fusion_weight_source": "fixed",
            }
        )
        config["feature_set"] = None
    else:
        metrics.update({"feature_set": feature_set, "classifier": "logistic"})
    (root / "dep_hc_task_diagnostic.json").write_text(
        json.dumps({"task": "dep_hc", "config": config, "metrics": metrics}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (root / "dep_hc_subject_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject_id",
                "cohort",
                "subject_score_p_dep",
                "predicted_cohort",
            ],
        )
        writer.writeheader()
        writer.writerow({"subject_id": "DEP001", "cohort": "DEP", "subject_score_p_dep": "0.2", "predicted_cohort": "HC"})
        writer.writerow({"subject_id": "HC001", "cohort": "HC", "subject_score_p_dep": "0.1", "predicted_cohort": "HC"})


def test_summarize_dep_hc_diagnostics_writes_board_and_hard_subjects(tmp_path: Path) -> None:
    _write_run(tmp_path / "single", feature_set="traditional")
    _write_run(tmp_path / "fusion", fusion=True)

    out_dir = tmp_path / "report"
    assert (
        summarize_dep_hc_diagnostics.main(
            [
                str(tmp_path),
                "--out-dir",
                str(out_dir),
                "--name",
                "board",
            ]
        )
        == 0
    )

    board_rows = list(csv.DictReader((out_dir / "board.csv").open(encoding="utf-8")))
    assert {row["feature_or_fusion"] for row in board_rows} == {
        "traditional",
        "traditional+time_frequency",
    }
    fusion_row = next(row for row in board_rows if row["classifier"] == "fusion")
    assert fusion_row["fusion_weight_source"] == "fixed"
    assert fusion_row["dep_vs_hc_ratio"] == "2.0000"

    hard_rows = list(csv.DictReader((out_dir / "board_hard_subjects.csv").open(encoding="utf-8")))
    assert hard_rows[0]["subject_id"] == "DEP001"
    assert hard_rows[0]["error_type"] == "false_negative"
    assert hard_rows[0]["n_errors"] == "2"
    assert hard_rows[0]["false_negative_errors"] == "2"
    assert hard_rows[0]["false_positive_errors"] == "0"
    assert "single" in hard_rows[0]["runs"]
    markdown = (out_dir / "board.md").read_text(encoding="utf-8")
    assert "## Robust Recommendation" in markdown
    assert "## Aggregate Summary" in markdown
    assert "| p2 | traditional+time_frequency | fixed_0_5/mean | fixed | holdouts=123 | 1 | 0.7500 | 0.7500 |  | 0.5000 | 1.0000 | 0.5000 | 0.5000 |" in markdown


def test_summarize_dep_hc_diagnostics_glob_only_expands_diagnostic_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_run(tmp_path / "outputs" / "dep_hc_a", feature_set="traditional")
    (tmp_path / "outputs" / "dep_hc_empty").mkdir(parents=True)
    (tmp_path / "outputs" / "dep_hc_notes.txt").write_text("not a run\n", encoding="utf-8")
    monkeypatch.setattr(summarize_dep_hc_diagnostics, "ROOT", tmp_path)

    runs = summarize_dep_hc_diagnostics.load_dep_hc_runs([Path("outputs/dep_hc_*")])

    assert [Path(run["root"]).name for run in runs] == ["dep_hc_a"]


def test_summarize_dep_hc_diagnostics_supports_absolute_glob(tmp_path: Path) -> None:
    _write_run(tmp_path / "dep_hc_abs", feature_set="traditional")

    runs = summarize_dep_hc_diagnostics.load_dep_hc_runs([tmp_path / "dep_hc_*"])

    assert [Path(run["root"]).name for run in runs] == ["dep_hc_abs"]


def test_summarize_dep_hc_diagnostics_normalizes_fusion_order_and_zero_hc_ratio(tmp_path: Path) -> None:
    _write_run(
        tmp_path / "fusion_a",
        fusion=True,
        fusion_features=["traditional", "time_frequency"],
        subject_ba=0.5,
        hc_recall=0.0,
        dep_recall=1.0,
    )
    _write_run(
        tmp_path / "fusion_b",
        fusion=True,
        fusion_features=["time_frequency", "traditional"],
        subject_ba=0.75,
        hc_recall=0.5,
        dep_recall=1.0,
    )

    rows = [summarize_dep_hc_diagnostics.board_row(run) for run in summarize_dep_hc_diagnostics.load_dep_hc_runs([tmp_path])]
    assert {row["feature_or_fusion"] for row in rows} == {"traditional+time_frequency"}
    assert rows[0]["dep_vs_hc_ratio"] == "inf"
    aggregate = summarize_dep_hc_diagnostics.aggregate_rows(rows)
    assert len(aggregate) == 1
    assert aggregate[0]["n"] == "2"


def test_summarize_dep_hc_diagnostics_labels_neural_models(tmp_path: Path) -> None:
    root = tmp_path / "dep_hc_neural"
    root.mkdir(parents=True)
    payload = {
        "task": "dep_hc",
        "config": {"protocol": "p2", "split_id": "split_h123", "model_name": "deformer_lite"},
        "metrics": {
            "task": "dep_hc",
            "model_name": "deformer_lite",
            "subject_ba": 0.75,
            "hc_subject_recall": 0.5,
            "dep_subject_recall": 1.0,
            "threshold_objective": "balanced_accuracy",
            "threshold_source": "validation_subjects",
        },
    }
    (root / "dep_hc_task_diagnostic.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (root / "dep_hc_subject_metrics.csv").write_text(
        "subject_id,cohort,subject_score_p_dep,predicted_cohort\nDEP001,DEP,0.8,DEP\n",
        encoding="utf-8",
    )

    run = summarize_dep_hc_diagnostics.load_dep_hc_runs([root])[0]
    row = summarize_dep_hc_diagnostics.board_row(run)

    assert row["feature_or_fusion"] == "neural:deformer_lite"
    assert row["classifier"] == "neural"


def test_summarize_dep_hc_diagnostics_labels_score_fusion_and_robust_rank(tmp_path: Path) -> None:
    root = tmp_path / "dep_hc_score_fusion"
    root.mkdir(parents=True)
    payload = {
        "task": "dep_hc",
        "config": {"protocol": "p2", "split_id": "split_h999"},
        "metrics": {
            "task": "dep_hc",
            "score_fusion_components": ["deformer", "traditional_tf"],
            "score_fusion_weights_by_component": {"deformer": 0.7, "traditional_tf": 0.3},
            "fusion_weight_source": "fixed",
            "subject_ba": 0.75,
            "hc_subject_recall": 0.5,
            "dep_subject_recall": 1.0,
            "threshold_objective": "fixed_0_5",
            "threshold_source": "fixed_0.5",
            "subject_aggregation": "vote_frac",
            "crop_combo_status": "computed",
            "crop_combo_expected_ba": 0.8,
            "crop_combo_worst_ba": 0.625,
            "crop_combo_expected_hc_recall": 0.75,
            "crop_combo_expected_dep_recall": 0.85,
        },
    }
    (root / "dep_hc_task_diagnostic.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (root / "dep_hc_subject_metrics.csv").write_text(
        "subject_id,cohort,subject_score_p_dep,predicted_cohort\nDEP001,DEP,0.8,DEP\n",
        encoding="utf-8",
    )

    run = summarize_dep_hc_diagnostics.load_dep_hc_runs([root])[0]
    row = summarize_dep_hc_diagnostics.board_row(run)
    aggregate = summarize_dep_hc_diagnostics.aggregate_rows([row])
    robust = summarize_dep_hc_diagnostics.robust_recommendation_rows(aggregate)

    assert row["feature_or_fusion"] == "score:deformer+traditional_tf"
    assert row["classifier"] == "score_fusion"
    assert row["primary_min_recall"] == "0.7500"
    assert row["primary_recall_gap"] == "0.1000"
    assert aggregate[0]["mean_worst_combo_ba"] == "0.6250"
    assert robust[0]["feature_or_fusion"] == "score:deformer+traditional_tf"


def test_summarize_dep_hc_diagnostics_normalizes_score_fusion_component_order(tmp_path: Path) -> None:
    for run_id, components in {
        "score_a": ["deformer", "traditional_tf"],
        "score_b": ["traditional_tf", "deformer"],
    }.items():
        root = tmp_path / run_id
        root.mkdir(parents=True)
        payload = {
            "task": "dep_hc",
            "config": {"protocol": "p2", "split_id": "split_h999"},
            "metrics": {
                "task": "dep_hc",
                "score_fusion_components": components,
                "score_fusion_weights_by_component": {"deformer": 0.7, "traditional_tf": 0.3},
                "fusion_weight_source": "fixed",
                "subject_ba": 0.75,
                "hc_subject_recall": 0.5,
                "dep_subject_recall": 1.0,
                "threshold_objective": "fixed_0_5",
                "threshold_source": "fixed_0.5",
                "subject_aggregation": "mean",
            },
        }
        (root / "dep_hc_task_diagnostic.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
        (root / "dep_hc_subject_metrics.csv").write_text(
            "subject_id,cohort,subject_score_p_dep,predicted_cohort\nDEP001,DEP,0.8,DEP\n",
            encoding="utf-8",
        )

    rows = [summarize_dep_hc_diagnostics.board_row(run) for run in summarize_dep_hc_diagnostics.load_dep_hc_runs([tmp_path])]
    aggregate = summarize_dep_hc_diagnostics.aggregate_rows(rows)

    assert {row["feature_or_fusion"] for row in rows} == {"score:deformer+traditional_tf"}
    assert len(aggregate) == 1
    assert aggregate[0]["n"] == "2"


def test_summarize_dep_hc_diagnostics_preserves_config_eval_scope(tmp_path: Path) -> None:
    for run_id, scope in {
        "p3_inner": "p3_inner_validation",
        "p3_outer": "p3_outer_test",
    }.items():
        root = tmp_path / run_id
        root.mkdir(parents=True)
        payload = {
            "task": "dep_hc",
            "config": {"protocol": "p3", "eval_scope": scope, "split_id": f"split_{scope}"},
            "metrics": {
                "task": "dep_hc",
                "model_name": "deformer_lite",
                "subject_ba": 0.75,
                "hc_subject_recall": 0.5,
                "dep_subject_recall": 1.0,
                "threshold_objective": "fixed_0_5",
                "threshold_source": "fixed_0.5",
                "subject_aggregation": "mean",
            },
        }
        (root / "dep_hc_task_diagnostic.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
        (root / "dep_hc_subject_metrics.csv").write_text(
            "subject_id,cohort,subject_score_p_dep,predicted_cohort\nDEP001,DEP,0.8,DEP\n",
            encoding="utf-8",
        )

    rows = [summarize_dep_hc_diagnostics.board_row(run) for run in summarize_dep_hc_diagnostics.load_dep_hc_runs([tmp_path])]
    aggregate = summarize_dep_hc_diagnostics.aggregate_rows(rows)

    assert {row["eval_scope"] for row in rows} == {"p3_inner_validation", "p3_outer_test"}
    assert {row["eval_scope"] for row in aggregate} == {"p3_inner_validation", "p3_outer_test"}


def test_robust_recommendation_rows_do_not_promote_missing_recall_gap() -> None:
    complete = {
        "eval_scope": "p2",
        "feature_or_fusion": "complete",
        "threshold_objective": "balanced_accuracy",
        "subject_aggregation": "vote_frac",
        "split_coverage": "holdouts=123",
        "n": "1",
        "mean_primary_ba": "0.7000",
        "mean_worst_combo_ba": "0.5000",
        "mean_min_recall": "0.6000",
        "mean_recall_gap": "0.1000",
    }
    missing_gap = {
        **complete,
        "feature_or_fusion": "missing_gap",
        "mean_primary_ba": "0.9000",
        "mean_worst_combo_ba": "0.9000",
        "mean_min_recall": "0.9000",
        "mean_recall_gap": "",
    }

    robust = summarize_dep_hc_diagnostics.robust_recommendation_rows([missing_gap, complete])

    assert robust[0]["feature_or_fusion"] == "complete"


def test_holdout_seed_from_config_supports_existing_split_id_format() -> None:
    assert (
        summarize_dep_hc_diagnostics._holdout_seed_from_config(
            {"split_id": "dep_hc_router_p2_holdout123_train42"}
        )
        == "123"
    )
    assert summarize_dep_hc_diagnostics._holdout_seed_from_config({"split_id": "split_h456"}) == "456"
