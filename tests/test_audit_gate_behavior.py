from pathlib import Path
import json

from hust_bci_er.audit.manifest import sha256_file
from scripts.audit_experiment import run_audit


ROUTE = Path("configs/routes/models/ea_deformer.yaml")


def test_candidate_gate_blocks_missing_run_artifacts():
    report = run_audit(ROUTE, None, gate="candidate")
    assert report["overall"] == "BLOCKED"


def test_smoke_gate_keeps_missing_run_as_warn():
    report = run_audit(ROUTE, None, gate="smoke")
    assert report["overall"] == "WARN"


def test_candidate_gate_blocks_placeholder_split_evidence():
    report = run_audit(ROUTE, None, gate="candidate")
    rules = {check["rule_id"]: check["status"] for check in report["checks"]}
    assert rules["SPLIT_MANIFEST_VERIFIABLE"] == "FAIL"


def test_no_top4_route_does_not_warn_when_top4_columns_are_absent(tmp_path):
    route = tmp_path / "no_top4_route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: no_top4_route",
                "status: IDEA",
                "dataset_version: train_v1",
                "split_id: p1_seed42_fold0",
                "seed: 42",
                "input_window_sec: 10",
                "preprocessing: [zscore]",
                "features: []",
                "model: {name: deformer_lite}",
                "adaptation: none",
                "inference: {top4: false, crop_policy: single}",
                "evaluation: {protocol: p1_repeated_group_kfold, primary_metric: exact_single_crop_expected_BA}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    snapshot = run_dir / "config_snapshot.yaml"
    snapshot.write_text(route.read_text(encoding="utf-8"), encoding="utf-8")
    prediction = run_dir / "predictions.csv"
    prediction.write_text(
        "subject_id,trial_id,y_score,y_pred,y_true\n"
        "s1,t1,0.1,0,0\n"
        "s1,t2,0.9,1,1\n",
        encoding="utf-8",
    )

    manifest = {
        "audit_schema_version": 1,
        "route_id": "no_top4_route",
        "git_commit": "test",
        "config_path": str(route),
        "config_snapshot_path": "config_snapshot.yaml",
        "config_sha256": sha256_file(snapshot),
        "dataset_manifest_path": "configs/datasets/train_v1.yaml",
        "dataset_manifest_sha256": sha256_file(Path("configs/datasets/train_v1.yaml")),
        "split_manifest_path": "configs/splits/p1_seed42_fold0.yaml",
        "split_id": "p1_seed42_fold0",
        "split_sha256": sha256_file(Path("configs/splits/p1_seed42_fold0.yaml")),
        "seed": 42,
        "command": "test",
        "primary_metric": "exact_single_crop_expected_BA",
        "metrics": {"exact_single_crop_expected_BA": 1.0},
        "prediction_csv": "predictions.csv",
        "prediction_sha256": sha256_file(prediction),
        "prediction_record_level": "trial",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(route, run_dir, gate="candidate")
    top4_checks = [check for check in report["checks"] if check["rule_id"] == "PREDICTION_TOP4_GROUPS"]
    assert top4_checks[-1]["status"] == "PASS"


def test_exact_primary_metric_is_recomputed_from_score_matrix(tmp_path):
    route = tmp_path / "exact_metric_route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: exact_metric_route",
                "status: IDEA",
                "dataset_version: train_v1",
                "split_id: p1_seed42_fold0",
                "seed: 42",
                "input_window_sec: 10",
                "preprocessing: [zscore]",
                "features: []",
                "model: {name: deformer_lite}",
                "adaptation: none",
                "inference: {top4: true, crop_policy: single}",
                "evaluation: {protocol: p1_repeated_group_kfold, primary_metric: exact_single_crop_expected_BA}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    snapshot = run_dir / "config_snapshot.yaml"
    snapshot.write_text(route.read_text(encoding="utf-8"), encoding="utf-8")

    predictions = run_dir / "predictions.csv"
    lines = ["subject_id,trial_id,y_score,y_pred,y_true,pred_top4"]
    for idx in range(8):
        label = 1 if idx < 4 else 0
        score = 1.0 - idx * 0.01
        lines.append(f"s1,t{idx},{score},{label},{label},{label}")
    predictions.write_text("\n".join(lines) + "\n", encoding="utf-8")

    matrix = run_dir / "score_matrix.csv"
    matrix_lines = ["subject_id,trial_id,y_true,crop_0,crop_1,crop_2,crop_3,crop_4"]
    for idx in range(8):
        label = 1 if idx < 4 else 0
        score = 1.0 - idx * 0.01
        matrix_lines.append(f"s1,t{idx},{label},{score},{score},{score},{score},{score}")
    matrix.write_text("\n".join(matrix_lines) + "\n", encoding="utf-8")

    manifest = {
        "audit_schema_version": 1,
        "route_id": "exact_metric_route",
        "git_commit": "test",
        "config_path": str(route),
        "config_snapshot_path": "config_snapshot.yaml",
        "config_sha256": sha256_file(snapshot),
        "dataset_manifest_path": "configs/datasets/train_v1.yaml",
        "dataset_manifest_sha256": sha256_file(Path("configs/datasets/train_v1.yaml")),
        "split_manifest_path": "configs/splits/p1_seed42_fold0.yaml",
        "split_id": "p1_seed42_fold0",
        "split_sha256": sha256_file(Path("configs/splits/p1_seed42_fold0.yaml")),
        "seed": 42,
        "command": "test",
        "primary_metric": "exact_single_crop_expected_BA",
        "metrics": {"exact_single_crop_expected_BA": 1.0},
        "prediction_csv": "predictions.csv",
        "prediction_sha256": sha256_file(predictions),
        "prediction_record_level": "trial",
        "top4_group_keys": ["subject_id"],
        "metric_inputs": {"score_matrix_csv": "score_matrix.csv"},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(route, run_dir, gate="candidate")
    rules = {check["rule_id"]: check["status"] for check in report["checks"]}
    assert rules["PRIMARY_METRIC_RECOMPUTE"] == "PASS"
    assert rules["PRIMARY_METRIC_REPORTED"] == "PASS"
    assert rules["PREDICTION_TOP4_RANKING"] == "PASS"


def test_manifest_path_escape_is_rejected(tmp_path):
    route = tmp_path / "escape_route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: escape_route",
                "status: IDEA",
                "dataset_version: train_v1",
                "split_id: p1_seed42_fold0",
                "seed: 42",
                "input_window_sec: 10",
                "preprocessing: [zscore]",
                "features: []",
                "model: {name: deformer_lite}",
                "adaptation: none",
                "inference: {top4: false, crop_policy: single}",
                "evaluation: {protocol: p1_repeated_group_kfold, primary_metric: no_top4_BA}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    snapshot = run_dir / "config_snapshot.yaml"
    snapshot.write_text(route.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = {
        "audit_schema_version": 1,
        "route_id": "escape_route",
        "git_commit": "test",
        "config_path": str(route),
        "config_snapshot_path": "../config_snapshot.yaml",
        "config_sha256": sha256_file(snapshot),
        "dataset_manifest_path": "configs/datasets/train_v1.yaml",
        "dataset_manifest_sha256": sha256_file(Path("configs/datasets/train_v1.yaml")),
        "split_manifest_path": "configs/splits/p1_seed42_fold0.yaml",
        "split_id": "p1_seed42_fold0",
        "split_sha256": sha256_file(Path("configs/splits/p1_seed42_fold0.yaml")),
        "seed": 42,
        "command": "test",
        "primary_metric": "no_top4_BA",
        "metrics": {"no_top4_BA": 1.0},
        "prediction_csv": "predictions.csv",
        "prediction_sha256": "0" * 64,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(route, run_dir, gate="candidate")
    manifest_check = [check for check in report["checks"] if check["rule_id"] == "MANIFEST_VALID"][-1]
    assert manifest_check["status"] == "FAIL"
    assert "run directory" in manifest_check["message"]
