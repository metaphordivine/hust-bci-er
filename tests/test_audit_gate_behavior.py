from pathlib import Path
import json

from hust_bci_er.audit.manifest import sha256_file
from scripts.audit_experiment import check_promotion_audit, dataset_checksum_errors, run_audit, split_leakage_errors


ROUTE = Path("configs/routes/models/ea_deformer.yaml")


def test_candidate_gate_blocks_missing_run_artifacts():
    report = run_audit(ROUTE, None, gate="candidate")
    assert report["overall"] == "BLOCKED"


def test_smoke_gate_keeps_missing_run_as_warn():
    report = run_audit(ROUTE, None, gate="smoke")
    assert report["overall"] == "WARN"


def test_candidate_gate_blocks_placeholder_run_split_evidence(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    snapshot = run_dir / "config_snapshot.yaml"
    snapshot.write_text(ROUTE.read_text(encoding="utf-8"), encoding="utf-8")
    prediction = run_dir / "predictions.csv"
    prediction.write_text(
        "subject_id,trial_id,y_score,y_pred,y_true,pred_top4\n"
        + "\n".join(f"s1,t{idx},{1.0 - idx * 0.01},{1 if idx < 4 else 0},{1 if idx < 4 else 0},{1 if idx < 4 else 0}" for idx in range(8))
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "audit_schema_version": 1,
        "route_id": "ea_deformer",
        "git_commit": "test",
        "config_path": str(ROUTE),
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
        "top4_group_keys": ["subject_id"],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(ROUTE, run_dir, gate="candidate")
    rules = {check["rule_id"]: check["status"] for check in report["checks"]}
    assert rules["RUN_SPLIT_EVIDENCE_VALID"] == "FAIL"


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
        "metric_inputs": {"score_matrix_csv": "score_matrix.csv", "score_matrix_sha256": sha256_file(matrix)},
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


def test_top4_ba_uses_pred_top4_for_primary_metric(tmp_path):
    route = tmp_path / "top4_metric_route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: top4_metric_route",
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
                "evaluation: {protocol: p1_repeated_group_kfold, primary_metric: top4_BA}",
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
    lines = ["subject_id,trial_id,y_score,y_pred,y_true,pred_top4"]
    for idx in range(8):
        label = 1 if idx < 4 else 0
        # y_pred is intentionally wrong; top4_BA must use pred_top4 instead.
        lines.append(f"s1,t{idx},{1.0 - idx * 0.01},{1 - label},{label},{label}")
    prediction.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "audit_schema_version": 1,
        "route_id": "top4_metric_route",
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
        "primary_metric": "top4_BA",
        "metrics": {"top4_BA": 1.0},
        "prediction_csv": "predictions.csv",
        "prediction_sha256": sha256_file(prediction),
        "prediction_record_level": "trial",
        "top4_group_keys": ["subject_id"],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(route, run_dir, gate="smoke")
    rules = {check["rule_id"]: check["status"] for check in report["checks"]}
    assert rules["PRIMARY_METRIC_RECOMPUTE"] == "PASS"
    assert rules["PRIMARY_METRIC_REPORTED"] == "PASS"


def test_bad_top4_value_is_structured_failure(tmp_path):
    route = tmp_path / "bad_top4_route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: bad_top4_route",
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
                "evaluation: {protocol: p1_repeated_group_kfold, primary_metric: top4_BA}",
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
    lines = ["subject_id,trial_id,y_score,y_pred,y_true,pred_top4"]
    for idx in range(8):
        label = 1 if idx < 4 else 0
        pred_top4 = "bad" if idx == 0 else str(label)
        lines.append(f"s1,t{idx},{1.0 - idx * 0.01},{label},{label},{pred_top4}")
    prediction.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "audit_schema_version": 1,
        "route_id": "bad_top4_route",
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
        "primary_metric": "top4_BA",
        "metrics": {"top4_BA": 1.0},
        "prediction_csv": "predictions.csv",
        "prediction_sha256": sha256_file(prediction),
        "prediction_record_level": "trial",
        "top4_group_keys": ["subject_id"],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(route, run_dir, gate="smoke")
    rules = {check["rule_id"]: check["status"] for check in report["checks"]}
    assert rules["PREDICTION_TOP4_BINARY"] == "FAIL"


def test_score_matrix_truth_balance_failure_is_structured(tmp_path):
    route = tmp_path / "bad_matrix_route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: bad_matrix_route",
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
    prediction = run_dir / "predictions.csv"
    lines = ["subject_id,trial_id,y_score,y_pred,y_true,pred_top4"]
    for idx in range(8):
        label = 1 if idx < 4 else 0
        lines.append(f"s1,t{idx},{1.0 - idx * 0.01},{label},{label},{label}")
    prediction.write_text("\n".join(lines) + "\n", encoding="utf-8")
    matrix = run_dir / "score_matrix.csv"
    matrix_lines = ["subject_id,trial_id,y_true,crop_0,crop_1,crop_2,crop_3,crop_4"]
    for idx in range(8):
        label = 1 if idx < 3 else 0
        score = 1.0 - idx * 0.01
        matrix_lines.append(f"s1,t{idx},{label},{score},{score},{score},{score},{score}")
    matrix.write_text("\n".join(matrix_lines) + "\n", encoding="utf-8")
    manifest = {
        "audit_schema_version": 1,
        "route_id": "bad_matrix_route",
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
        "top4_group_keys": ["subject_id"],
        "metric_inputs": {"score_matrix_csv": "score_matrix.csv", "score_matrix_sha256": sha256_file(matrix)},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(route, run_dir, gate="smoke")
    rules = {check["rule_id"]: check["status"] for check in report["checks"]}
    assert rules["SCORE_MATRIX_TRUTH_BALANCE"] == "FAIL"


def test_malformed_manifest_is_structured_failure(tmp_path):
    route = tmp_path / "malformed_manifest_route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: malformed_manifest_route",
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
    (run_dir / "manifest.json").write_text("{not-json", encoding="utf-8")

    report = run_audit(route, run_dir, gate="candidate")
    manifest_check = [check for check in report["checks"] if check["rule_id"] == "MANIFEST_VALID"][-1]
    assert manifest_check["status"] == "FAIL"
    assert "parse failed" in manifest_check["message"]


def test_run_specific_dataset_and_split_evidence_can_override_repo_placeholders(tmp_path):
    route = tmp_path / "run_evidence_route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: run_evidence_route",
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
    prediction = run_dir / "predictions.csv"
    prediction.write_text(
        "subject_id,trial_id,y_score,y_pred,y_true\n"
        "s1,t1,0.1,0,0\n"
        "s1,t2,0.9,1,1\n",
        encoding="utf-8",
    )
    dataset = run_dir / "dataset_manifest.yaml"
    dataset.write_text(
        "\n".join(
            [
                "dataset_version: train_v1",
                "status: ready",
                "data_sources:",
                "  - path: data/train.csv",
                "    kind: table",
                "checksum_manifest:",
                "  - path: data/train.csv",
                "    sha256: '" + "1" * 64 + "'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    split = run_dir / "split_manifest.yaml"
    split.write_text(
        "\n".join(
            [
                "split_id: p1_seed42_fold0",
                "subject_group_split: true",
                "status: ready",
                "train_subjects: [s1]",
                "val_subjects: [s2]",
                "test_subjects: [s3]",
                "trial_rows: []",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "audit_schema_version": 1,
        "route_id": "run_evidence_route",
        "git_commit": "test",
        "config_path": str(route),
        "config_snapshot_path": "config_snapshot.yaml",
        "config_sha256": sha256_file(snapshot),
        "dataset_manifest_path": "dataset_manifest.yaml",
        "dataset_manifest_sha256": sha256_file(dataset),
        "split_manifest_path": "split_manifest.yaml",
        "split_id": "p1_seed42_fold0",
        "split_sha256": sha256_file(split),
        "seed": 42,
        "command": "test",
        "primary_metric": "no_top4_BA",
        "metrics": {"no_top4_BA": 1.0},
        "prediction_csv": "predictions.csv",
        "prediction_sha256": sha256_file(prediction),
        "prediction_record_level": "trial",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(route, run_dir, gate="candidate")
    rules = {check["rule_id"]: check["status"] for check in report["checks"]}
    assert rules["RUN_DATASET_EVIDENCE_VALID"] == "PASS"
    assert rules["RUN_SPLIT_EVIDENCE_VALID"] == "PASS"
    assert rules["RUN_DATASET_CHECKSUM_SCHEMA"] == "PASS"
    assert rules["RUN_DATASET_CHECKSUM_COVERAGE"] == "PASS"
    assert rules["RUN_SPLIT_SUBJECT_DISJOINT"] == "PASS"
    assert rules["RUN_SPLIT_ORIGINAL_TRIAL_DISJOINT"] == "PASS"


def test_run_split_subject_overlap_is_structured_failure(tmp_path):
    route = tmp_path / "split_overlap_route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: split_overlap_route",
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
    prediction = run_dir / "predictions.csv"
    prediction.write_text("subject_id,trial_id,y_score,y_pred,y_true\ns1,t1,0.1,0,0\ns2,t2,0.9,1,1\n", encoding="utf-8")
    dataset = run_dir / "dataset_manifest.yaml"
    dataset.write_text(
        "dataset_version: train_v1\nstatus: ready\ndata_sources:\n  - path: data/train.csv\n    kind: table\nchecksum_manifest:\n  - path: data/train.csv\n    sha256: '"
        + "1" * 64
        + "'\n",
        encoding="utf-8",
    )
    split = run_dir / "split_manifest.yaml"
    split.write_text(
        "split_id: p1_seed42_fold0\nsubject_group_split: true\nstatus: ready\ntrain_subjects: [s1]\nval_subjects: [s1]\ntest_subjects: [s3]\ntrial_rows: []\n",
        encoding="utf-8",
    )
    manifest = {
        "audit_schema_version": 1,
        "route_id": "split_overlap_route",
        "git_commit": "test",
        "config_path": str(route),
        "config_snapshot_path": "config_snapshot.yaml",
        "config_sha256": sha256_file(snapshot),
        "dataset_manifest_path": "dataset_manifest.yaml",
        "dataset_manifest_sha256": sha256_file(dataset),
        "split_manifest_path": "split_manifest.yaml",
        "split_id": "p1_seed42_fold0",
        "split_sha256": sha256_file(split),
        "seed": 42,
        "command": "test",
        "primary_metric": "no_top4_BA",
        "metrics": {"no_top4_BA": 1.0},
        "prediction_csv": "predictions.csv",
        "prediction_sha256": sha256_file(prediction),
        "prediction_record_level": "trial",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(route, run_dir, gate="candidate")
    rules = {check["rule_id"]: check["status"] for check in report["checks"]}
    assert rules["RUN_SPLIT_SUBJECT_DISJOINT"] == "FAIL"


def test_run_split_original_trial_overlap_is_structured_failure(tmp_path):
    route = tmp_path / "trial_overlap_route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: trial_overlap_route",
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
    prediction = run_dir / "predictions.csv"
    prediction.write_text("subject_id,trial_id,y_score,y_pred,y_true\ns1,t1,0.1,0,0\ns2,t2,0.9,1,1\n", encoding="utf-8")
    dataset = run_dir / "dataset_manifest.yaml"
    dataset.write_text(
        "dataset_version: train_v1\nstatus: ready\ndata_sources:\n  - path: data/train.csv\n    kind: table\nchecksum_manifest:\n  - path: data/train.csv\n    sha256: '"
        + "1" * 64
        + "'\n",
        encoding="utf-8",
    )
    split = run_dir / "split_manifest.yaml"
    split.write_text(
        "\n".join(
            [
                "split_id: p1_seed42_fold0",
                "subject_group_split: true",
                "status: ready",
                "train_subjects: [s1]",
                "val_subjects: [s2]",
                "test_subjects: [s3]",
                "trial_rows:",
                "  - subject_id: s1",
                "    original_trial_id: t1",
                "    split: train",
                "  - subject_id: s1",
                "    original_trial_id: t1",
                "    split: test",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "audit_schema_version": 1,
        "route_id": "trial_overlap_route",
        "git_commit": "test",
        "config_path": str(route),
        "config_snapshot_path": "config_snapshot.yaml",
        "config_sha256": sha256_file(snapshot),
        "dataset_manifest_path": "dataset_manifest.yaml",
        "dataset_manifest_sha256": sha256_file(dataset),
        "split_manifest_path": "split_manifest.yaml",
        "split_id": "p1_seed42_fold0",
        "split_sha256": sha256_file(split),
        "seed": 42,
        "command": "test",
        "primary_metric": "no_top4_BA",
        "metrics": {"no_top4_BA": 1.0},
        "prediction_csv": "predictions.csv",
        "prediction_sha256": sha256_file(prediction),
        "prediction_record_level": "trial",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(route, run_dir, gate="candidate")
    rules = {check["rule_id"]: check["status"] for check in report["checks"]}
    assert rules["RUN_SPLIT_ORIGINAL_TRIAL_DISJOINT"] == "FAIL"


def test_dataset_checksum_coverage_and_schema_are_checked(tmp_path):
    route = tmp_path / "dataset_checksum_route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: dataset_checksum_route",
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
    prediction = run_dir / "predictions.csv"
    prediction.write_text("subject_id,trial_id,y_score,y_pred,y_true\ns1,t1,0.1,0,0\ns2,t2,0.9,1,1\n", encoding="utf-8")
    dataset = run_dir / "dataset_manifest.yaml"
    dataset.write_text(
        "\n".join(
            [
                "dataset_version: train_v1",
                "status: ready",
                "data_sources:",
                "  - path: data/a.csv",
                "    kind: table",
                "  - path: data/b.csv",
                "    kind: table",
                "checksum_manifest:",
                "  - path: data/a.csv",
                "    sha256: not-a-sha",
                "",
            ]
        ),
        encoding="utf-8",
    )
    split = run_dir / "split_manifest.yaml"
    split.write_text(
        "split_id: p1_seed42_fold0\nsubject_group_split: true\nstatus: ready\ntrain_subjects: [s1]\nval_subjects: [s2]\ntest_subjects: [s3]\ntrial_rows: []\n",
        encoding="utf-8",
    )
    manifest = {
        "audit_schema_version": 1,
        "route_id": "dataset_checksum_route",
        "git_commit": "test",
        "config_path": str(route),
        "config_snapshot_path": "config_snapshot.yaml",
        "config_sha256": sha256_file(snapshot),
        "dataset_manifest_path": "dataset_manifest.yaml",
        "dataset_manifest_sha256": sha256_file(dataset),
        "split_manifest_path": "split_manifest.yaml",
        "split_id": "p1_seed42_fold0",
        "split_sha256": sha256_file(split),
        "seed": 42,
        "command": "test",
        "primary_metric": "no_top4_BA",
        "metrics": {"no_top4_BA": 1.0},
        "prediction_csv": "predictions.csv",
        "prediction_sha256": sha256_file(prediction),
        "prediction_record_level": "trial",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(route, run_dir, gate="candidate")
    rules = {check["rule_id"]: check["status"] for check in report["checks"]}
    assert rules["RUN_DATASET_CHECKSUM_SCHEMA"] == "FAIL"
    assert rules["RUN_DATASET_CHECKSUM_COVERAGE"] == "FAIL"


def test_top4_truth_balance_message_is_explicit_when_y_true_is_absent(tmp_path):
    route = tmp_path / "top4_no_truth_route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: top4_no_truth_route",
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
                "evaluation: {protocol: p1_repeated_group_kfold, primary_metric: top4_BA}",
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
    lines = ["subject_id,trial_id,y_score,pred_top4"]
    for idx in range(8):
        label = 1 if idx < 4 else 0
        lines.append(f"s1,t{idx},{1.0 - idx * 0.01},{label}")
    prediction.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "audit_schema_version": 1,
        "route_id": "top4_no_truth_route",
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
        "primary_metric": "top4_BA",
        "metrics": {"top4_BA": 1.0},
        "prediction_csv": "predictions.csv",
        "prediction_sha256": sha256_file(prediction),
        "prediction_record_level": "trial",
        "top4_group_keys": ["subject_id"],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(route, run_dir, gate="smoke")
    checks = {check["rule_id"]: check for check in report["checks"]}
    assert checks["PREDICTION_TOP4_TRUTH_BALANCE"]["status"] == "PASS"
    assert "not checked" in checks["PREDICTION_TOP4_TRUTH_BALANCE"]["message"]


def test_top4_ranking_is_warn_when_score_is_absent(tmp_path):
    route = tmp_path / "top4_no_score_route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: top4_no_score_route",
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
                "evaluation: {protocol: p1_repeated_group_kfold, primary_metric: top4_BA}",
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
    lines = ["subject_id,trial_id,y_true,pred_top4"]
    for idx in range(8):
        label = 1 if idx < 4 else 0
        lines.append(f"s1,t{idx},{label},{label}")
    prediction.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "audit_schema_version": 1,
        "route_id": "top4_no_score_route",
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
        "primary_metric": "top4_BA",
        "metrics": {"top4_BA": 1.0},
        "prediction_csv": "predictions.csv",
        "prediction_sha256": sha256_file(prediction),
        "prediction_record_level": "trial",
        "top4_group_keys": ["subject_id"],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(route, run_dir, gate="smoke")
    rules = {check["rule_id"]: check["status"] for check in report["checks"]}
    assert rules["PREDICTION_SCORE_NUMERIC"] == "WARN"
    assert rules["PREDICTION_TOP4_RANKING"] == "WARN"


def test_trial_rows_also_enforce_subject_disjointness():
    subject_errors, trial_errors = split_leakage_errors(
        {
            "trial_rows": [
                {"subject_id": "s1", "original_trial_id": "t1", "split": "train"},
                {"subject_id": "s1", "original_trial_id": "t2", "split": "test"},
            ]
        }
    )
    assert subject_errors
    assert not trial_errors


def test_trial_rows_enforce_subject_disjointness_per_fold():
    subject_errors, _ = split_leakage_errors(
        {
            "trial_rows": [
                {"fold": 0, "subject_id": "s1", "original_trial_id": "t1", "split": "train"},
                {"fold": 0, "subject_id": "s1", "original_trial_id": "t2", "split": "test"},
                {"fold": 1, "subject_id": "s1", "original_trial_id": "t3", "split": "train"},
                {"fold": 1, "subject_id": "s2", "original_trial_id": "t4", "split": "test"},
            ]
        }
    )
    assert any("fold 0" in item for item in subject_errors)
    assert not any("fold 1" in item for item in subject_errors)


def test_dataset_checksum_extra_paths_are_reported():
    schema_errors, coverage_errors, extra_errors = dataset_checksum_errors(
        {
            "data_sources": [{"path": "data/a.csv", "kind": "table"}],
            "checksum_manifest": [
                {"path": "data/a.csv", "sha256": "1" * 64},
                {"path": "data/old.csv", "sha256": "2" * 64},
            ],
        }
    )
    assert not schema_errors
    assert not coverage_errors
    assert extra_errors


def test_promotion_audit_rejects_minimal_candidate_report(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.audit_experiment.ROOT", tmp_path)
    audit = tmp_path / "reports" / "audits" / "candidate.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({"route_id": "summary_route", "gate": "candidate", "overall": "PASS"}), encoding="utf-8")
    promotion = tmp_path / "reports" / "promotion_audits" / "summary_route_promotion.md"
    promotion.parent.mkdir(parents=True)
    promotion.write_text(
        "\n".join(
            [
                "route_id: summary_route",
                "promoted_from_run: outputs/summary_route/run",
                "candidate_audit_report: reports/audits/candidate.json",
                "primary_metric: top4_BA",
                "comparison_baseline: baseline",
                "risk_review: reviewed",
                "no_leakage_review: reviewed",
                "decision: promote",
                "reviewer: test",
                "date: 2026-05-14",
                "",
            ]
        ),
        encoding="utf-8",
    )
    checks = []
    check_promotion_audit("summary_route", promotion, checks)
    rules = {check["rule_id"]: check["status"] for check in checks}
    assert rules["PROMOTION_AUDIT_CANDIDATE_REPORT"] == "PASS"
    assert rules["PROMOTION_AUDIT_CANDIDATE_RULES"] == "FAIL"
