from __future__ import annotations

import csv
import json
from pathlib import Path

from hust_bci_er.analysis.candidate_scores import analyze_score_matrices, render_markdown_report


def _write_run(run_dir: Path, route_id: str, *, offset: float = 0.0, reverse: bool = False) -> None:
    run_dir.mkdir(parents=True)
    rows = []
    for subject in ("DEP001", "HC001"):
        for idx in range(8):
            y_true = 1 if idx < 4 else 0
            base = 0.85 - idx * 0.08 if not reverse else 0.25 + idx * 0.08
            rows.append(
                {
                    "subject_id": subject,
                    "trial_id": f"{subject}_t{idx}",
                    "y_true": str(y_true),
                    "crop_0": f"{base + offset:.6f}",
                    "crop_1": f"{base + offset + 0.01:.6f}",
                    "crop_2": f"{base + offset + 0.02:.6f}",
                    "crop_3": f"{base + offset + 0.03:.6f}",
                    "crop_4": f"{base + offset + 0.04:.6f}",
                }
            )
    with (run_dir / "score_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "route_id": route_id,
                "primary_metric": "exact_single_crop_expected_BA",
                "top4_group_keys": ["subject_id"],
            }
        ),
        encoding="utf-8",
    )


def test_analyze_candidate_scores_builds_markdown_sections(tmp_path):
    _write_run(tmp_path / "good", "good_route")
    _write_run(tmp_path / "bad", "bad_route", reverse=True)

    analysis = analyze_score_matrices([tmp_path])
    report = render_markdown_report(analysis)

    assert {item.route_id for item in analysis.diagnostics} == {"good_route", "bad_route"}
    assert analysis.diagnostics[0].n_groups == 2
    assert "# Candidate Score Deep Dive" in report
    assert "## Crop Performance" in report
    assert "## Score Correlation" in report
    assert "## Frequent Error Samples" in report
    assert "`bad_route`" in report


def test_analyze_candidate_scores_reports_pairwise_disagreement(tmp_path):
    _write_run(tmp_path / "good", "good_route")
    _write_run(tmp_path / "bad", "bad_route", reverse=True)

    analysis = analyze_score_matrices([tmp_path])

    assert analysis.disagreement_matrix["good_route"]["good_route"] == 0.0
    assert analysis.disagreement_matrix["good_route"]["bad_route"] == 1.0
    assert analysis.error_samples
    assert analysis.error_samples[0]["wrong_models"] == 1
