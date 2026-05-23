from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts import summarize_dep_hc_diagnostics


def _write_run(
    root: Path,
    *,
    feature_set: str | None = "traditional",
    fusion: bool = False,
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
        metrics.update(
            {
                "fusion_feature_sets": ["traditional", "time_frequency"],
                "fusion_weight_by_feature": {"traditional": 0.5, "time_frequency": 0.5},
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
    assert hard_rows[0]["n_errors"] == "2"
    assert "single" in hard_rows[0]["runs"]
    markdown = (out_dir / "board.md").read_text(encoding="utf-8")
    assert "## Aggregate Summary" in markdown
    assert "| p2 | traditional+time_frequency | fixed | 1 | 0.7500 | 0.7500 | 0.5000 | 1.0000 |" in markdown
