import csv
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from hust_bci_er.audit.manifest import sha256_file
from scripts import export_component_scores, run_score_fusion_routes


ROUTE = Path("configs/routes/models/conformer_srfnet_score_average.yaml")


def _write_score_matrix(
    path: Path,
    *,
    base: float,
    repeated_source_crop: int | None = None,
    source_crops: list[int] | None = None,
) -> None:
    if repeated_source_crop is not None and source_crops is not None:
        raise ValueError("set repeated_source_crop or source_crops, not both")
    crop_sources = source_crops
    if repeated_source_crop is not None:
        crop_sources = [repeated_source_crop for _idx in range(5)]
    header = [
        "subject_id",
        "trial_id",
        "y_true",
    ]
    for crop in range(5):
        header.append(f"crop_{crop}")
        if crop_sources is not None:
            header.extend([f"crop_{crop}_source_crop_id", f"crop_{crop}_window_start_sec"])
    rows = []
    for idx in range(8):
        label = 1 if idx >= 4 else 0
        scores = [base + idx * 0.1 + crop * 0.01 for crop in range(5)]
        row = ["s1", f"t{idx}", str(label)]
        for crop, score in enumerate(scores):
            row.append(f"{score:.3f}")
            if crop_sources is not None:
                row.extend([str(crop_sources[crop]), f"{float(crop_sources[crop]):.8f}"])
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_export_component_scores_prefers_score_matrix_crops(tmp_path):
    matrix = tmp_path / "score_matrix.csv"
    output = tmp_path / "component.csv"
    _write_score_matrix(matrix, base=0.0)

    rc = export_component_scores.export_component_scores_from_score_matrix(
        matrix,
        "conformer_component",
        output,
    )

    assert rc == 0
    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    assert len(rows) == 40
    assert rows[0]["component_id"] == "conformer_component"
    assert {row["crop_id"] for row in rows[:5]} == {"0", "1", "2", "3", "4"}


def test_export_component_scores_rejects_partial_score_matrix_provenance_columns(tmp_path, capsys):
    matrix = tmp_path / "score_matrix.csv"
    output = tmp_path / "component.csv"
    header = ["subject_id", "trial_id", "y_true", *[f"crop_{idx}" for idx in range(5)], "crop_0_source_crop_id"]
    matrix.write_text(
        ",".join(header) + "\n" + "s1,t0,1,0.1,0.2,0.3,0.4,0.5,0\n",
        encoding="utf-8",
    )

    rc = export_component_scores.export_component_scores_from_score_matrix(
        matrix,
        "conformer_component",
        output,
    )

    assert rc == 1
    assert "partial crop provenance columns" in capsys.readouterr().err
    assert not output.exists()


def test_export_component_scores_rejects_partial_prediction_provenance(tmp_path, capsys):
    predictions = tmp_path / "predictions.csv"
    output = tmp_path / "component.csv"
    predictions.write_text(
        "subject_id,trial_id,source_crop_id,window_start_sec,score\n"
        "s1,t0,,,0.1\n"
        "s1,t1,4,4.00000000,0.2\n",
        encoding="utf-8",
    )

    rc = export_component_scores.export_component_scores(
        predictions,
        "conformer_component",
        output,
    )

    assert rc == 1
    assert "partial crop provenance values" in capsys.readouterr().err
    assert not output.exists()


def test_score_fusion_runner_rejects_partial_score_matrix_provenance_columns(tmp_path):
    matrix = tmp_path / "score_matrix.csv"
    header = ["subject_id", "trial_id", "y_true", *[f"crop_{idx}" for idx in range(5)], "crop_0_source_crop_id"]
    matrix.write_text(
        ",".join(header) + "\n" + "s1,t0,1,0.1,0.2,0.3,0.4,0.5,0\n",
        encoding="utf-8",
    )
    source = run_score_fusion_routes.SourceArtifact(matrix, "score_matrix")

    try:
        run_score_fusion_routes._component_rows_from_score_matrix(source, "conformer_component", "sliding_window_conformer_lite")
    except ValueError as exc:
        assert "partial crop provenance columns" in str(exc)
    else:
        raise AssertionError("partial score-matrix provenance columns should fail")


def test_score_fusion_export_rejects_single_prediction_provenance_column(tmp_path, capsys):
    predictions = tmp_path / "predictions.csv"
    output = tmp_path / "component.csv"
    predictions.write_text(
        "subject_id,trial_id,source_crop_id,score\n"
        "s1,t0,0,0.1\n",
        encoding="utf-8",
    )
    source = run_score_fusion_routes.SourceArtifact(predictions, "predictions")

    rc = run_score_fusion_routes.export_component_score(
        "conformer_component",
        "ea_deformer",
        [source],
        output,
    )

    assert rc == 1
    assert "partial crop provenance columns" in capsys.readouterr().err
    assert not output.exists()


def test_export_component_scores_rejects_empty_single_prediction_provenance_column(tmp_path, capsys):
    predictions = tmp_path / "predictions.csv"
    output = tmp_path / "component.csv"
    predictions.write_text(
        "subject_id,trial_id,source_crop_id,score\n"
        "s1,t0,,0.1\n",
        encoding="utf-8",
    )

    rc = export_component_scores.export_component_scores(
        predictions,
        "conformer_component",
        output,
    )

    assert rc == 1
    assert "partial crop provenance columns" in capsys.readouterr().err
    assert not output.exists()


def test_score_fusion_runner_writes_prediction_matrix_metric_and_manifest(tmp_path):
    conformer_matrix = tmp_path / "conformer_score_matrix.csv"
    srfnet_matrix = tmp_path / "srfnet_score_matrix.csv"
    conformer_component = tmp_path / "conformer_component.csv"
    srfnet_component = tmp_path / "srfnet_component.csv"
    _write_score_matrix(conformer_matrix, base=0.0)
    _write_score_matrix(srfnet_matrix, base=1.0)
    assert export_component_scores.export_component_scores_from_score_matrix(
        conformer_matrix,
        "conformer_component",
        conformer_component,
    ) == 0
    assert export_component_scores.export_component_scores_from_score_matrix(
        srfnet_matrix,
        "srfnet_long_component",
        srfnet_component,
    ) == 0

    output_dir = tmp_path / "run"
    rc, info = run_score_fusion_routes.assemble_score_fusion(
        ROUTE,
        {
            "conformer_component": conformer_component,
            "srfnet_long_component": srfnet_component,
        },
        output_dir,
    )
    assert rc == 0
    run_score_fusion_routes._write_score_fusion_manifest(
        ROUTE,
        output_dir,
        {
            "conformer_component": conformer_component,
            "srfnet_long_component": srfnet_component,
        },
        source_manifest_paths=[],
        score_matrix_evidence=info["score_matrix_evidence"],
        component_score_evidence={"conformer_component": "genuine", "srfnet_long_component": "genuine"},
    )

    prediction_rows = list(csv.DictReader((output_dir / "predictions.csv").open(encoding="utf-8", newline="")))
    matrix_rows = list(csv.DictReader((output_dir / "score_matrix.csv").open(encoding="utf-8", newline="")))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert len(prediction_rows) == 8
    assert len(matrix_rows) == 8
    assert "y_pred" not in prediction_rows[0]
    assert "crop_4" in matrix_rows[0]
    assert (output_dir / "metric_audit.json").exists()
    assert manifest["metric_inputs"]["score_matrix_csv"] == "score_matrix.csv"
    assert manifest["score_matrix_evidence"] == "genuine"
    assert {item["component_id"] for item in manifest["component_score_inputs"]} == {
        "conformer_component",
        "srfnet_long_component",
    }


def test_score_fusion_preserves_primary_component_crop_provenance(tmp_path):
    conformer_matrix = tmp_path / "conformer_score_matrix.csv"
    srfnet_matrix = tmp_path / "srfnet_score_matrix.csv"
    conformer_component = tmp_path / "conformer_component.csv"
    srfnet_component = tmp_path / "srfnet_component.csv"
    _write_score_matrix(conformer_matrix, base=0.0, repeated_source_crop=0)
    _write_score_matrix(srfnet_matrix, base=1.0, repeated_source_crop=0)
    assert export_component_scores.export_component_scores_from_score_matrix(
        conformer_matrix,
        "conformer_component",
        conformer_component,
    ) == 0
    assert export_component_scores.export_component_scores_from_score_matrix(
        srfnet_matrix,
        "srfnet_long_component",
        srfnet_component,
    ) == 0

    output_dir = tmp_path / "run"
    rc, _info = run_score_fusion_routes.assemble_score_fusion(
        ROUTE,
        {
            "conformer_component": conformer_component,
            "srfnet_long_component": srfnet_component,
        },
        output_dir,
    )

    assert rc == 0
    matrix_rows = list(csv.DictReader((output_dir / "score_matrix.csv").open(encoding="utf-8", newline="")))
    assert matrix_rows[0]["crop_4_source_crop_id"] == "0"
    assert matrix_rows[0]["crop_4_window_start_sec"] == "0.00000000"


def test_score_fusion_preserves_per_crop_source_provenance(tmp_path):
    conformer_matrix = tmp_path / "conformer_score_matrix.csv"
    srfnet_matrix = tmp_path / "srfnet_score_matrix.csv"
    conformer_component = tmp_path / "conformer_component.csv"
    srfnet_component = tmp_path / "srfnet_component.csv"
    source_crops = [10, 11, 12, 13, 14]
    _write_score_matrix(conformer_matrix, base=0.0, source_crops=source_crops)
    _write_score_matrix(srfnet_matrix, base=1.0, source_crops=source_crops)
    assert export_component_scores.export_component_scores_from_score_matrix(
        conformer_matrix,
        "conformer_component",
        conformer_component,
    ) == 0
    assert export_component_scores.export_component_scores_from_score_matrix(
        srfnet_matrix,
        "srfnet_long_component",
        srfnet_component,
    ) == 0

    output_dir = tmp_path / "run"
    rc, info = run_score_fusion_routes.assemble_score_fusion(
        ROUTE,
        {
            "conformer_component": conformer_component,
            "srfnet_long_component": srfnet_component,
        },
        output_dir,
    )

    assert rc == 0
    run_score_fusion_routes._write_score_fusion_manifest(
        ROUTE,
        output_dir,
        {
            "conformer_component": conformer_component,
            "srfnet_long_component": srfnet_component,
        },
        source_manifest_paths=[],
        score_matrix_evidence=info["score_matrix_evidence"],
        component_score_evidence={"conformer_component": "genuine", "srfnet_long_component": "genuine"},
    )
    matrix_rows = list(csv.DictReader((output_dir / "score_matrix.csv").open(encoding="utf-8", newline="")))
    assert matrix_rows[0]["crop_0_source_crop_id"] == "10"
    assert matrix_rows[0]["crop_0_window_start_sec"] == "10.00000000"
    assert matrix_rows[0]["crop_4_source_crop_id"] == "14"
    assert matrix_rows[0]["crop_4_window_start_sec"] == "14.00000000"
    dataset = yaml.safe_load((output_dir / "dataset_manifest.yaml").read_text(encoding="utf-8"))
    assert dataset["trial_index"][0]["crop_ids"] == ["10", "11", "12", "13", "14"]
    assert dataset["trial_index"][0]["window_start_secs"] == [10.0, 11.0, 12.0, 13.0, 14.0]


def _write_source_manifest(
    root: Path,
    *,
    seed: int,
    fold: int,
    include_seed_fold: bool = True,
    score_matrix_evidence: str = "genuine",
    crop_policy: dict | None = None,
) -> Path:
    root.mkdir(parents=True)
    split = {
        "split_id": f"split_seed{seed}_fold{fold}",
        "train_subjects": [f"train{fold}"],
        "val_subjects": [],
        "test_subjects": ["s1"],
        "trial_rows": [
            {"subject_id": f"train{fold}", "trial_id": "t0", "original_trial_id": f"train{fold}::t0", "split": "train"},
            *[
                {"subject_id": "s1", "trial_id": f"t{idx}", "original_trial_id": f"s1::t{idx}", "split": "test"}
                for idx in range(8)
            ],
        ],
    }
    split_path = root / "split.yaml"
    split_path.write_text(yaml.safe_dump(split, sort_keys=False), encoding="utf-8")
    manifest = {
        "split_id": split["split_id"],
        "split_manifest_path": "split.yaml",
        "split_sha256": sha256_file(split_path),
        "score_matrix_evidence": score_matrix_evidence,
    }
    if crop_policy is not None:
        manifest["crop_policy"] = crop_policy
        manifest["protocol_job_crop_policy"] = crop_policy
    if include_seed_fold:
        manifest["seed"] = seed
        manifest["fold"] = fold
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _write_base_run(root: Path, route_id: str, *, seed: int, fold: int, base: float, score_matrix_evidence: str = "genuine") -> None:
    run_dir = root / route_id
    _write_source_manifest(run_dir, seed=seed, fold=fold, score_matrix_evidence=score_matrix_evidence)
    _write_score_matrix(run_dir / "score_matrix.csv", base=base)


def test_score_fusion_manifest_merges_multiple_source_splits(tmp_path):
    conformer_matrix = tmp_path / "conformer_score_matrix.csv"
    srfnet_matrix = tmp_path / "srfnet_score_matrix.csv"
    conformer_component = tmp_path / "conformer_component.csv"
    srfnet_component = tmp_path / "srfnet_component.csv"
    _write_score_matrix(conformer_matrix, base=0.0)
    _write_score_matrix(srfnet_matrix, base=1.0)
    assert export_component_scores.export_component_scores_from_score_matrix(conformer_matrix, "conformer_component", conformer_component) == 0
    assert export_component_scores.export_component_scores_from_score_matrix(srfnet_matrix, "srfnet_long_component", srfnet_component) == 0
    source_a = _write_source_manifest(tmp_path / "source_a", seed=42, fold=0)
    source_a_dup = _write_source_manifest(tmp_path / "source_a_dup", seed=42, fold=0)
    source_b = _write_source_manifest(tmp_path / "source_b", seed=42, fold=1)
    output_dir = tmp_path / "run"
    rc, info = run_score_fusion_routes.assemble_score_fusion(
        ROUTE,
        {"conformer_component": conformer_component, "srfnet_long_component": srfnet_component},
        output_dir,
    )
    assert rc == 0

    run_score_fusion_routes._write_score_fusion_manifest(
        ROUTE,
        output_dir,
        {"conformer_component": conformer_component, "srfnet_long_component": srfnet_component},
        source_manifest_paths=[source_a, source_a_dup, source_b],
        score_matrix_evidence=info["score_matrix_evidence"],
        component_score_evidence={"conformer_component": "genuine", "srfnet_long_component": "genuine"},
    )

    split = yaml.safe_load((output_dir / "split_manifest.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert split["status"] == "merged_source_protocol_splits"
    assert {fold["fold"] for fold in split["fold_definitions"]} == {"seed42_fold0", "seed42_fold1"}
    assert {str(row["fold"]) for row in split["trial_rows"]} == {"seed42_fold0", "seed42_fold1"}
    assert len(split["fold_definitions"]) == 2
    assert manifest["score_fusion_source_evidence"]["split_evidence"] == "merged_source_protocol_splits"


def test_score_fusion_manifest_inherits_common_source_crop_policy(tmp_path):
    conformer_matrix = tmp_path / "conformer_score_matrix.csv"
    srfnet_matrix = tmp_path / "srfnet_score_matrix.csv"
    conformer_component = tmp_path / "conformer_component.csv"
    srfnet_component = tmp_path / "srfnet_component.csv"
    _write_score_matrix(conformer_matrix, base=0.0, repeated_source_crop=0)
    _write_score_matrix(srfnet_matrix, base=1.0, repeated_source_crop=0)
    assert export_component_scores.export_component_scores_from_score_matrix(conformer_matrix, "conformer_component", conformer_component) == 0
    assert export_component_scores.export_component_scores_from_score_matrix(srfnet_matrix, "srfnet_long_component", srfnet_component) == 0
    source_policy = {"name": "crop1", "selection": "fixed_index", "crop_index": 0, "tie_break": "not_applicable"}
    source_a = _write_source_manifest(tmp_path / "source_a", seed=42, fold=0, crop_policy=source_policy)
    source_b = _write_source_manifest(tmp_path / "source_b", seed=42, fold=0, crop_policy=source_policy)
    output_dir = tmp_path / "run"
    rc, info = run_score_fusion_routes.assemble_score_fusion(
        ROUTE,
        {"conformer_component": conformer_component, "srfnet_long_component": srfnet_component},
        output_dir,
    )
    assert rc == 0

    run_score_fusion_routes._write_score_fusion_manifest(
        ROUTE,
        output_dir,
        {"conformer_component": conformer_component, "srfnet_long_component": srfnet_component},
        source_manifest_paths=[source_a, source_b],
        score_matrix_evidence=info["score_matrix_evidence"],
        component_score_evidence={"conformer_component": "genuine", "srfnet_long_component": "genuine"},
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["crop_policy"] == source_policy


def test_score_fusion_manifest_keeps_source_split_when_dataset_is_missing(tmp_path):
    conformer_matrix = tmp_path / "conformer_score_matrix.csv"
    srfnet_matrix = tmp_path / "srfnet_score_matrix.csv"
    conformer_component = tmp_path / "conformer_component.csv"
    srfnet_component = tmp_path / "srfnet_component.csv"
    _write_score_matrix(conformer_matrix, base=0.0, repeated_source_crop=0)
    _write_score_matrix(srfnet_matrix, base=1.0, repeated_source_crop=0)
    assert export_component_scores.export_component_scores_from_score_matrix(conformer_matrix, "conformer_component", conformer_component) == 0
    assert export_component_scores.export_component_scores_from_score_matrix(srfnet_matrix, "srfnet_long_component", srfnet_component) == 0
    source_a = _write_source_manifest(tmp_path / "source_a", seed=42, fold=0)
    source_b = _write_source_manifest(tmp_path / "source_b", seed=42, fold=0)
    for source in (source_a, source_b):
        manifest = json.loads(source.read_text(encoding="utf-8"))
        manifest.pop("dataset_manifest_path", None)
        source.write_text(json.dumps(manifest), encoding="utf-8")
    output_dir = tmp_path / "run"
    rc, info = run_score_fusion_routes.assemble_score_fusion(
        ROUTE,
        {"conformer_component": conformer_component, "srfnet_long_component": srfnet_component},
        output_dir,
    )
    assert rc == 0

    run_score_fusion_routes._write_score_fusion_manifest(
        ROUTE,
        output_dir,
        {"conformer_component": conformer_component, "srfnet_long_component": srfnet_component},
        source_manifest_paths=[source_a, source_b],
        score_matrix_evidence=info["score_matrix_evidence"],
        component_score_evidence={"conformer_component": "genuine", "srfnet_long_component": "genuine"},
    )

    split = yaml.safe_load((output_dir / "split_manifest.yaml").read_text(encoding="utf-8"))
    dataset = yaml.safe_load((output_dir / "dataset_manifest.yaml").read_text(encoding="utf-8"))
    assert split["train_subjects"]
    assert split["test_subjects"]
    assert dataset["status"] == "ready"


def test_score_fusion_manifest_uses_protocol_job_fold_metadata(tmp_path):
    protocol_root = tmp_path / "protocol"
    job_dir = protocol_root / "job_runs" / "p1__route__seed42__fold1"
    source = _write_source_manifest(job_dir, seed=42, fold=1, include_seed_fold=False)
    (protocol_root / "protocol_run_manifest.json").write_text(
        json.dumps({"jobs": [{"job_id": job_dir.name, "seed": 42, "fold": 1}]}),
        encoding="utf-8",
    )

    split_path = run_score_fusion_routes._write_merged_source_split(
        {"split_id": "fusion_split"},
        tmp_path / "fusion",
        [source],
    )

    split = yaml.safe_load(split_path.read_text(encoding="utf-8"))
    assert split["fold_definitions"][0]["fold"] == "seed42_fold1"
    assert {row["fold"] for row in split["trial_rows"]} == {"seed42_fold1"}


def test_score_fusion_main_trusts_component_csvs_exported_earlier_in_same_run(tmp_path, monkeypatch):
    base_runs = tmp_path / "base_runs"
    _write_base_run(base_runs, "sliding_window_conformer_lite", seed=42, fold=0, base=0.0)
    _write_base_run(base_runs, "sliding_window_srfnet", seed=42, fold=0, base=1.0)
    _write_base_run(base_runs, "fixed_crop_ea_fbstcnet", seed=42, fold=0, base=2.0)
    _write_base_run(base_runs, "sliding_ea_dgcnn_dann_cohort_w6_s1", seed=42, fold=0, base=3.0)
    _write_base_run(base_runs, "sliding_ea_dgcnn_coral_cohort_w6_s1", seed=42, fold=0, base=4.0)
    _write_base_run(base_runs, "fixed_crop_car_fbstcnet", seed=42, fold=0, base=5.0)
    _write_base_run(base_runs, "fixed_crop_whitening_eps3e4_fbstcnet", seed=42, fold=0, base=6.0)
    _write_base_run(base_runs, "sliding_window_srfnet_whitening_eps3e4", seed=42, fold=0, base=7.0)

    audit_calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        audit_calls.append([str(part) for part in cmd])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_score_fusion_routes.subprocess, "run", fake_run)

    output_dir = tmp_path / "score_fusion"
    rc = run_score_fusion_routes.main(
        [
            "--route-filter", "score_average",
            "--base-runs-dir", str(base_runs),
            "--export-missing",
            "--output-dir", str(output_dir),
            "--audit",
        ]
    )

    summary = json.loads((output_dir / "score_fusion_summary.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert summary["passed"] == 5
    assert [item["status"] for item in summary["results"]] == ["PASS", "PASS", "PASS", "PASS", "PASS"]
    assert sum("repo_doctor.py" in call for cmd in audit_calls for call in cmd) == 5


def _write_protocol_job_base_run(root: Path, route_id: str, *, suffixes: list[str], base: float) -> None:
    run_dir = root / route_id
    jobs = []
    for offset, suffix in enumerate(suffixes):
        job_id = f"p2__{route_id}__{suffix}"
        job_dir = run_dir / "job_runs" / job_id
        _write_source_manifest(job_dir, seed=42, fold=0)
        _write_score_matrix(job_dir / "score_matrix.csv", base=base + offset)
        jobs.append({"job_id": job_id, "seed": 42, "fold": 0})
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "protocol_run_manifest.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")


def _write_p2_crop_policy_base_run(root: Path, route_id: str, *, base: float, worst_source_crop: int) -> None:
    run_dir = root / route_id
    jobs = []
    suffixes = [f"eval_crop{idx}" for idx in range(1, 6)] + ["eval_worst"]
    for offset, suffix in enumerate(suffixes):
        job_id = f"p2__{route_id}__{suffix}"
        job_dir = run_dir / "job_runs" / job_id
        repeated_source_crop = int(suffix[-1]) - 1 if suffix.startswith("eval_crop") else worst_source_crop
        if suffix.startswith("eval_crop"):
            crop_policy = {
                "name": f"crop{repeated_source_crop + 1}",
                "selection": "fixed_index",
                "crop_index": repeated_source_crop,
                "tie_break": "not_applicable",
            }
        else:
            crop_policy = {
                "name": "worst",
                "selection": "label_aware_min_metric_stress_test",
                "tie_break": "lowest_assignment_index",
            }
        _write_source_manifest(job_dir, seed=42, fold=0, crop_policy=crop_policy)
        _write_score_matrix(job_dir / "score_matrix.csv", base=base + offset, repeated_source_crop=repeated_source_crop)
        jobs.append({"job_id": job_id, "seed": 42, "fold": 0})
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "protocol_run_manifest.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")


def test_score_fusion_main_keeps_protocol_jobs_distinct(tmp_path):
    base_runs = tmp_path / "base_runs"
    suffixes = ["eval_crop1", "eval_crop2"]
    _write_protocol_job_base_run(base_runs, "sliding_window_conformer_lite", suffixes=suffixes, base=0.0)
    _write_protocol_job_base_run(base_runs, "sliding_window_srfnet", suffixes=suffixes, base=1.0)

    output_dir = tmp_path / "score_fusion"
    rc = run_score_fusion_routes.main(
        [
            "--route-filter", "conformer_srfnet_score_average",
            "--base-runs-dir", str(base_runs),
            "--export-missing",
            "--output-dir", str(output_dir),
        ]
    )

    summary = json.loads((output_dir / "score_fusion_summary.json").read_text(encoding="utf-8"))
    prediction_rows = list(csv.DictReader((output_dir / "conformer_srfnet_score_average" / "predictions.csv").open(encoding="utf-8", newline="")))
    assert rc == 0
    assert summary["passed"] == 1
    assert len(prediction_rows) == 16
    assert {row["protocol_job"] for row in prediction_rows} == {"p2__eval_crop1", "p2__eval_crop2"}


def test_score_fusion_synthesizes_p2_worst_after_fusion(tmp_path):
    base_runs = tmp_path / "base_runs"
    _write_p2_crop_policy_base_run(base_runs, "sliding_window_conformer_lite", base=0.0, worst_source_crop=3)
    _write_p2_crop_policy_base_run(base_runs, "sliding_window_srfnet", base=1.0, worst_source_crop=4)

    output_dir = tmp_path / "score_fusion"
    rc = run_score_fusion_routes.main(
        [
            "--route-filter", "conformer_srfnet_score_average",
            "--base-runs-dir", str(base_runs),
            "--export-missing",
            "--output-dir", str(output_dir),
        ]
    )

    summary = json.loads((output_dir / "score_fusion_summary.json").read_text(encoding="utf-8"))
    component_rows = list(
        csv.DictReader((output_dir / "component_scores" / "conformer_component.csv").open(encoding="utf-8", newline=""))
    )
    prediction_rows = list(
        csv.DictReader(
            (output_dir / "conformer_srfnet_score_average" / "predictions.csv").open(encoding="utf-8", newline="")
        )
    )
    assert rc == 0
    assert summary["passed"] == 1
    assert "p2__eval_worst" not in {row["protocol_job"] for row in component_rows}
    assert "p2__eval_worst" in {row["protocol_job"] for row in prediction_rows}
    assert sum(row["protocol_job"] == "p2__eval_worst" for row in prediction_rows) == 8


def test_score_fusion_protocol_job_filter_exports_one_p2_crop_policy(tmp_path):
    base_runs = tmp_path / "base_runs"
    _write_p2_crop_policy_base_run(base_runs, "sliding_window_conformer_lite", base=0.0, worst_source_crop=3)
    _write_p2_crop_policy_base_run(base_runs, "sliding_window_srfnet", base=1.0, worst_source_crop=4)

    output_dir = tmp_path / "score_fusion"
    rc = run_score_fusion_routes.main(
        [
            "--route-filter", "conformer_srfnet_score_average",
            "--base-runs-dir", str(base_runs),
            "--export-missing",
            "--protocol-job-filter", "p2__eval_crop3",
            "--output-dir", str(output_dir),
        ]
    )

    summary = json.loads((output_dir / "score_fusion_summary.json").read_text(encoding="utf-8"))
    component_rows = list(
        csv.DictReader(
            (output_dir / "component_scores" / "p2__eval_crop3" / "conformer_component.csv").open(encoding="utf-8", newline="")
        )
    )
    prediction_rows = list(
        csv.DictReader(
            (output_dir / "conformer_srfnet_score_average" / "predictions.csv").open(encoding="utf-8", newline="")
        )
    )
    matrix_rows = list(
        csv.DictReader(
            (output_dir / "conformer_srfnet_score_average" / "score_matrix.csv").open(encoding="utf-8", newline="")
        )
    )
    manifest = json.loads((output_dir / "conformer_srfnet_score_average" / "manifest.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert summary["passed"] == 1
    assert {row["protocol_job"] for row in component_rows} == {"p2__eval_crop3"}
    assert {row["protocol_job"] for row in prediction_rows} == {"p2__eval_crop3"}
    assert {row["crop_0_source_crop_id"] for row in matrix_rows} == {"2"}
    assert manifest["crop_policy"]["name"] == "crop3"
    assert manifest["crop_policy"]["crop_index"] == 2


def test_score_fusion_synthesizes_p2_worst_without_truth_labels():
    rows = []
    for trial_idx in range(8):
        for crop_idx in range(5):
            rows.append(
                {
                    "route_id": "conformer_srfnet_score_average",
                    "protocol_job": f"p2__eval_crop{crop_idx + 1}",
                    "subject_id": "s1",
                    "trial_id": f"t{trial_idx}",
                    "crop_id": "0",
                    "score": str(float(5 - crop_idx + trial_idx)),
                    "pred_top4": "0",
                }
            )

    worst_rows = run_score_fusion_routes._synthesize_p2_worst_rows(rows)

    assert len(worst_rows) == 40
    assert {row["protocol_job"] for row in worst_rows} == {"p2__eval_worst"}
    assert {row["score"] for row in worst_rows if row["trial_id"] == "t0"} == {"1.0"}


def test_export_component_score_rejects_only_route_dependent_p2_worst(tmp_path, capsys):
    matrix = tmp_path / "score_matrix.csv"
    output = tmp_path / "component.csv"
    _write_score_matrix(matrix, base=0.0, repeated_source_crop=3)
    source = run_score_fusion_routes.SourceArtifact(
        matrix,
        "score_matrix",
        job_id="p2__sliding_window_conformer_lite__eval_worst",
    )

    rc = run_score_fusion_routes.export_component_score(
        "conformer_component",
        "sliding_window_conformer_lite",
        [source],
        output,
    )

    assert rc == 1
    assert "no component score rows after filtering" in capsys.readouterr().err
    assert not output.exists()


def test_source_protocol_job_key_strips_exact_base_route_id():
    source = run_score_fusion_routes.SourceArtifact(
        Path("score_matrix.csv"),
        "score_matrix",
        job_id="p2__route__with__parts__eval_crop1",
    )

    assert run_score_fusion_routes._source_protocol_job_key(source, "route__with__parts") == "p2__eval_crop1"


def test_score_fusion_audit_blocks_synthetic_component_evidence(tmp_path, monkeypatch):
    base_runs = tmp_path / "base_runs"
    _write_base_run(base_runs, "sliding_window_conformer_lite", seed=42, fold=0, base=0.0, score_matrix_evidence="synthetic")
    _write_base_run(base_runs, "sliding_window_srfnet", seed=42, fold=0, base=1.0)

    audit_calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        audit_calls.append([str(part) for part in cmd])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_score_fusion_routes.subprocess, "run", fake_run)

    output_dir = tmp_path / "score_fusion"
    rc = run_score_fusion_routes.main(
        [
            "--route-filter", "conformer_srfnet_score_average",
            "--base-runs-dir", str(base_runs),
            "--export-missing",
            "--output-dir", str(output_dir),
            "--audit",
        ]
    )

    summary = json.loads((output_dir / "score_fusion_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "conformer_srfnet_score_average" / "manifest.json").read_text(encoding="utf-8"))
    assert rc == 1
    assert summary["results"][0]["status"] == "AUDIT_BLOCKED_SYNTHETIC_COMPONENT_EVIDENCE"
    assert manifest["score_matrix_evidence"] == "synthetic"
    assert not any("repo_doctor.py" in call for cmd in audit_calls for call in cmd)
