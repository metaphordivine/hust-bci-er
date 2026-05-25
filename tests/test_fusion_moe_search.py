from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("pandas")

from hust_bci_er.analysis.fusion_moe_search import align_score_runs, load_score_run, run_posthoc_search


def _write_matrix(path: Path, *, include_metadata: bool, score_shift: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    if include_metadata:
        fields.extend(["protocol_job", "seed", "fold"])
    fields.extend(["subject_id", "trial_id", "y_true", "crop_0", "crop_1", "crop_2", "crop_3", "crop_4"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx in range(8):
            truth = 1 if idx >= 4 else 0
            base = float(idx) + score_shift
            row = {
                "subject_id": "DEP1001",
                "trial_id": f"DEP1001_trial{idx}",
                "y_true": str(truth),
                "crop_0": f"{base:.3f}",
                "crop_1": f"{base + 0.1:.3f}",
                "crop_2": f"{base + 0.2:.3f}",
                "crop_3": f"{base + 0.3:.3f}",
                "crop_4": f"{base + 0.4:.3f}",
            }
            if include_metadata:
                row.update({"protocol_job": "p1__seed42__fold0", "seed": "42", "fold": "0"})
            writer.writerow(row)


def test_expert_path_metadata_aligns_with_assembled_matrix(tmp_path: Path) -> None:
    assembled = tmp_path / "assembled" / "score_matrix.csv"
    expert = tmp_path / "job_runs" / "p1__some_route__seed42__fold0" / "score_matrix.csv"
    _write_matrix(assembled, include_metadata=True)
    _write_matrix(expert, include_metadata=False, score_shift=0.5)

    runs = [
        load_score_run("assembled", [assembled]),
        load_score_run("expert", [expert]),
    ]
    aligned = align_score_runs(runs)

    assert len(aligned.keys) == 8
    assert len(aligned.groups) == 1
    assert aligned.labels == ("assembled", "expert")


def test_protocol_job_key_disambiguates_duplicate_trial_rows(tmp_path: Path) -> None:
    first_crop0 = tmp_path / "first" / "job_runs" / "p2__first_route__crop0__seed42__fold0" / "score_matrix.csv"
    first_crop1 = tmp_path / "first" / "job_runs" / "p2__first_route__crop1__seed42__fold0" / "score_matrix.csv"
    second_crop0 = tmp_path / "second" / "job_runs" / "p2__second_route__crop0__seed42__fold0" / "score_matrix.csv"
    second_crop1 = tmp_path / "second" / "job_runs" / "p2__second_route__crop1__seed42__fold0" / "score_matrix.csv"
    _write_matrix(first_crop0, include_metadata=False)
    _write_matrix(first_crop1, include_metadata=False, score_shift=0.1)
    _write_matrix(second_crop0, include_metadata=False, score_shift=0.2)
    _write_matrix(second_crop1, include_metadata=False, score_shift=0.3)

    aligned = align_score_runs(
        [
            load_score_run("first", [first_crop0, first_crop1]),
            load_score_run("second", [second_crop0, second_crop1]),
        ]
    )

    assert len(aligned.keys) == 16
    assert len(aligned.groups) == 2


def test_posthoc_search_runs_on_tiny_aligned_matrices(tmp_path: Path) -> None:
    first = tmp_path / "first" / "score_matrix.csv"
    second = tmp_path / "job_runs" / "p1__second__seed42__fold0" / "score_matrix.csv"
    _write_matrix(first, include_metadata=True)
    _write_matrix(second, include_metadata=False, score_shift=0.25)
    aligned = align_score_runs([load_score_run("first", [first]), load_score_run("second", [second])])

    results = run_posthoc_search(
        aligned,
        workers=1,
        max_base_runs=2,
        screen_assignments=8,
        exact_top_k=2,
        fusion_asha_candidates=12,
        fusion_asha_min_groups=1,
    )

    assert not results["single_screen"].empty
    assert not results["fusion_exact"].empty
    assert not results["fusion_asha_exact"].empty
    assert not results["moe_exact"].empty


def test_posthoc_search_handles_empty_candidate_limits(tmp_path: Path) -> None:
    first = tmp_path / "first" / "score_matrix.csv"
    second = tmp_path / "job_runs" / "p1__second__seed42__fold0" / "score_matrix.csv"
    _write_matrix(first, include_metadata=True)
    _write_matrix(second, include_metadata=False, score_shift=0.25)
    aligned = align_score_runs([load_score_run("first", [first]), load_score_run("second", [second])])

    results = run_posthoc_search(
        aligned,
        workers=1,
        max_base_runs=0,
        screen_assignments=8,
        exact_top_k=0,
        fusion_asha_candidates=0,
    )

    assert results["fusion_exact"].empty
    assert results["fusion_asha_exact"].empty
    assert results["moe_exact"].empty
