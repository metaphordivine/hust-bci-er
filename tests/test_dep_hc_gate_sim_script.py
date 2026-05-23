from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts import simulate_dep_hc_subject_gate


def _write_run(root: Path, *, subject_ba: float = 0.75, hc_recall: float = 0.75, dep_recall: float = 0.75) -> None:
    root.mkdir(parents=True)
    payload = {
        "task": "dep_hc",
        "config": {"protocol": "p2", "split_id": root.name},
        "metrics": {
            "subject_ba": subject_ba,
            "hc_subject_recall": hc_recall,
            "dep_subject_recall": dep_recall,
        },
    }
    (root / "dep_hc_task_diagnostic.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    rows = [
        {"subject_id": "HC001", "cohort": "HC", "predicted_cohort": "HC", "subject_score_p_dep": "0.1"},
        {"subject_id": "HC002", "cohort": "HC", "predicted_cohort": "DEP", "subject_score_p_dep": "0.8"},
        {"subject_id": "DEP001", "cohort": "DEP", "predicted_cohort": "DEP", "subject_score_p_dep": "0.9"},
        {"subject_id": "DEP002", "cohort": "DEP", "predicted_cohort": "DEP", "subject_score_p_dep": "0.5"},
    ]
    with (root / "dep_hc_subject_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_dep_hc_subject_gate_sim_reports_bucket_precision_and_failures(tmp_path: Path) -> None:
    _write_run(tmp_path / "run_a")
    out_dir = tmp_path / "report"

    assert (
        simulate_dep_hc_subject_gate.main(
            [
                str(tmp_path / "run_a"),
                "--out-dir",
                str(out_dir),
                "--name",
                "gate",
                "--min-bucket-precision",
                "0.75",
            ]
        )
        == 0
    )

    run_rows = list(csv.DictReader((out_dir / "gate.csv").open(encoding="utf-8")))
    assert run_rows[0]["n_hc_like"] == "1"
    assert run_rows[0]["n_dep_like"] == "2"
    assert run_rows[0]["n_ambiguous"] == "1"
    assert run_rows[0]["dep_like_precision"] == "0.5000"
    assert run_rows[0]["passes_gate"] == "false"
    assert "dep_like_precision" in run_rows[0]["failed_reasons"]

    agg_rows = list(csv.DictReader((out_dir / "gate_aggregate.csv").open(encoding="utf-8")))
    assert agg_rows[0]["passes_gate"] == "false"
    assert (out_dir / "gate.md").exists()
