from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts import audit_dep_hc_subject_confidence


def _write_run(root: Path) -> None:
    root.mkdir(parents=True)
    payload = {
        "task": "dep_hc",
        "config": {"protocol": "p2", "split_id": "split_h123"},
        "metrics": {"subject_ba": 0.5},
    }
    (root / "dep_hc_task_diagnostic.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    rows = [
        {
            "subject_id": "DEP001",
            "cohort": "DEP",
            "predicted_cohort": "HC",
            "subject_score_p_dep": "0.1",
            "confidence": "0.9",
            "correct": "0",
        },
        {
            "subject_id": "HC001",
            "cohort": "HC",
            "predicted_cohort": "HC",
            "subject_score_p_dep": "0.52",
            "confidence": "0.52",
            "correct": "1",
        },
        {
            "subject_id": "HC002",
            "cohort": "HC",
            "predicted_cohort": "DEP",
            "subject_score_p_dep": "0.7",
            "confidence": "0.7",
            "correct": "0",
        },
    ]
    with (root / "dep_hc_subject_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_dep_hc_confidence_audit_reports_high_confidence_errors(tmp_path: Path) -> None:
    _write_run(tmp_path / "run")
    out_dir = tmp_path / "report"

    assert (
        audit_dep_hc_subject_confidence.main(
            [
                str(tmp_path / "run"),
                "--out-dir",
                str(out_dir),
                "--name",
                "audit",
                "--high-confidence",
                "0.75",
            ]
        )
        == 0
    )

    run_rows = list(csv.DictReader((out_dir / "audit.csv").open(encoding="utf-8")))
    assert run_rows[0]["n_errors"] == "2"
    assert run_rows[0]["n_high_conf_errors"] == "1"
    assert run_rows[0]["n_ambiguous"] == "1"

    subject_rows = list(csv.DictReader((out_dir / "audit_subjects.csv").open(encoding="utf-8")))
    assert subject_rows[0]["subject_id"] == "DEP001"
    assert subject_rows[0]["bucket"] == "high_conf_error"
    assert (out_dir / "audit.md").exists()
