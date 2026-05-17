import csv
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from hust_bci_er.audit.manifest import sha256_file
from scripts import export_component_scores, run_score_fusion_routes


ROUTE = Path("configs/routes/models/conformer_srfnet_score_average.yaml")


def _write_score_matrix(path: Path, *, base: float) -> None:
    header = [
        "subject_id",
        "trial_id",
        "y_true",
        "crop_0",
        "crop_1",
        "crop_2",
        "crop_3",
        "crop_4",
    ]
    rows = []
    for idx in range(8):
        label = 1 if idx >= 4 else 0
        scores = [base + idx * 0.1 + crop * 0.01 for crop in range(5)]
        rows.append(["s1", f"t{idx}", str(label), *[f"{score:.3f}" for score in scores]])
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
    )

    prediction_rows = list(csv.DictReader((output_dir / "predictions.csv").open(encoding="utf-8", newline="")))
    matrix_rows = list(csv.DictReader((output_dir / "score_matrix.csv").open(encoding="utf-8", newline="")))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert len(prediction_rows) == 8
    assert len(matrix_rows) == 8
    assert "crop_4" in matrix_rows[0]
    assert (output_dir / "metric_audit.json").exists()
    assert manifest["metric_inputs"]["score_matrix_csv"] == "score_matrix.csv"
    assert manifest["score_matrix_evidence"] == "genuine"
    assert {item["component_id"] for item in manifest["component_score_inputs"]} == {
        "conformer_component",
        "srfnet_long_component",
    }


def _write_source_manifest(root: Path, *, seed: int, fold: int) -> Path:
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
        "seed": seed,
        "fold": fold,
        "split_id": split["split_id"],
        "split_manifest_path": "split.yaml",
        "split_sha256": sha256_file(split_path),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _write_base_run(root: Path, route_id: str, *, seed: int, fold: int, base: float) -> None:
    run_dir = root / route_id
    _write_source_manifest(run_dir, seed=seed, fold=fold)
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
    )

    split = yaml.safe_load((output_dir / "split_manifest.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert split["status"] == "merged_source_protocol_splits"
    assert {fold["fold"] for fold in split["fold_definitions"]} == {"seed42_fold0", "seed42_fold1"}
    assert {str(row["fold"]) for row in split["trial_rows"]} == {"seed42_fold0", "seed42_fold1"}
    assert len(split["fold_definitions"]) == 2
    assert manifest["score_fusion_source_evidence"]["split_evidence"] == "merged_source_protocol_splits"


def test_score_fusion_main_trusts_component_csvs_exported_earlier_in_same_run(tmp_path, monkeypatch):
    base_runs = tmp_path / "base_runs"
    _write_base_run(base_runs, "sliding_window_conformer_lite", seed=42, fold=0, base=0.0)
    _write_base_run(base_runs, "sliding_window_srfnet", seed=42, fold=0, base=1.0)
    _write_base_run(base_runs, "fixed_crop_ea_fbstcnet", seed=42, fold=0, base=2.0)

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
    assert summary["passed"] == 3
    assert [item["status"] for item in summary["results"]] == ["PASS", "PASS", "PASS"]
    assert sum("repo_doctor.py" in call for cmd in audit_calls for call in cmd) == 3
