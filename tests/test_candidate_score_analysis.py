from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from hust_bci_er.analysis.candidate_scores import analyze_score_matrices, load_score_matrix_run, render_markdown_report


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


def _write_score_rows(run_dir: Path, route_id: str, rows: list[dict[str, str]]) -> None:
    run_dir.mkdir(parents=True)
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
    assert "## Mean-score Top-4 Disagreement" in report
    assert "## Frequent Error Samples" in report
    assert "`bad_route`" in report


def test_candidate_score_analysis_public_exports():
    from hust_bci_er.analysis import analyze_score_matrices as public_analyze

    assert public_analyze is analyze_score_matrices


def test_analyze_candidate_scores_reports_exact_cohort_ba(tmp_path):
    rows = []
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    correct = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]
    reverse = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    for subject in ("DEP001", "HC001"):
        for idx, label in enumerate(y_true):
            if subject.startswith("DEP"):
                scores = [correct[idx], reverse[idx], reverse[idx], reverse[idx], correct[idx]]
            else:
                scores = [correct[idx]] * 5
            rows.append(
                {
                    "subject_id": subject,
                    "trial_id": f"{subject}_t{idx}",
                    "y_true": str(label),
                    "crop_0": f"{scores[0]:.6f}",
                    "crop_1": f"{scores[1]:.6f}",
                    "crop_2": f"{scores[2]:.6f}",
                    "crop_3": f"{scores[3]:.6f}",
                    "crop_4": f"{scores[4]:.6f}",
                }
            )
    _write_score_rows(tmp_path / "cohort", "cohort_route", rows)

    analysis = analyze_score_matrices([tmp_path / "cohort"])
    diagnostic = analysis.diagnostics[0]
    report = render_markdown_report(analysis)

    assert diagnostic.mean_score_top4_ba == 0.5
    assert diagnostic.dep_ba == pytest.approx(0.4540672)
    assert diagnostic.hc_ba == pytest.approx(1.0)
    assert "DEP exact BA" in report


def test_manifest_group_keys_must_exist_in_score_matrix(tmp_path):
    run_dir = tmp_path / "missing_group_key"
    _write_run(run_dir, "missing_group_key_route")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "route_id": "missing_group_key_route",
                "primary_metric": "exact_single_crop_expected_BA",
                "top4_group_keys": ["subject_id", "fold"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing manifest group key"):
        load_score_matrix_run(run_dir / "score_matrix.csv")


def test_score_matrix_load_rejects_duplicate_trial_ids(tmp_path):
    run_dir = tmp_path / "duplicate_trial"
    _write_run(run_dir, "duplicate_trial_route")
    rows = list(csv.DictReader((run_dir / "score_matrix.csv").open(encoding="utf-8", newline="")))
    rows[1]["trial_id"] = rows[0]["trial_id"]
    with (run_dir / "score_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="unique trial_id"):
        load_score_matrix_run(run_dir / "score_matrix.csv")


def test_analyze_candidate_scores_reports_pairwise_disagreement(tmp_path):
    _write_run(tmp_path / "good", "good_route")
    _write_run(tmp_path / "bad", "bad_route", reverse=True)

    analysis = analyze_score_matrices([tmp_path])

    assert analysis.disagreement_matrix["good_route"]["good_route"] == 0.0
    assert analysis.disagreement_matrix["good_route"]["bad_route"] == 1.0
    assert analysis.error_samples
    assert analysis.error_samples[0]["wrong_models"] == 1
