from pathlib import Path
import csv
import json
import subprocess

import yaml

from hust_bci_er.audit.manifest import sha256_file
from hust_bci_er.contracts.prediction import prediction_schema
from hust_bci_er.evaluation.report import build_metric_report
from scripts.audit_experiment import (
    add_raw_data_source_dataset_match,
    check_prediction_scope_against_split,
    check_promotion_audit,
    check_reproducibility_manifest,
    check_run_gate_eligibility,
    check_evidence_lineage,
    check_route_model_kwargs_passthrough,
    compare_metric,
    dataset_checksum_errors,
    dataset_label_scope_errors,
    recompute_primary_metric_from_predictions,
    recompute_exact_metric_from_matrix,
    run_audit,
    split_evidence_consistency_errors,
    split_manifest_has_formal_evidence,
    split_leakage_errors,
)


ROUTE = Path("configs/routes/models/ea_deformer.yaml")
TORCH_ROUTE_DATA = {"training": {"job_adapter": "torch_classifier"}}


def reproducibility_metadata():
    return {
        "environment": {
            "python": {"version": "3.13.11"},
            "packages": {"numpy": "2.4.2", "PyYAML": "6.0.3", "torch": "2.12.0.dev20260408+cu128"},
            "torch": {"version": "2.12.0.dev20260408+cu128", "cuda": "12.8", "cudnn": 92000},
        },
        "determinism": {
            "python_seed": 42,
            "python_hash_seed": 42,
            "pythonhashseed_env": "42",
            "numpy_seed": 42,
            "torch_seed": 42,
            "deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "dataloader_worker_seed_base": 42,
            "batch_order": {"policy": "seeded_sampler_or_shuffle_false", "sampler_seed": 42},
        },
        "checkpoint_selection": {
            "rule": "best_monitored_epoch",
            "monitor": "val_loss",
            "mode": "min",
            "tie_break": "earliest_epoch",
            "restore_best": True,
        },
        "crop_policy": {"name": "single", "selection": "route_default_single_crop", "crop_index": 0, "tie_break": "not_applicable"},
        "environment_lock": {
            "files": [
                {"path": "environment.lock", "sha256": sha256_file(Path("environment.lock"))},
                {"path": "requirements.lock", "sha256": sha256_file(Path("requirements.lock"))},
            ]
        },
    }


def test_candidate_gate_blocks_missing_run_artifacts():
    report = run_audit(ROUTE, None, gate="candidate")
    assert report["overall"] == "BLOCKED"


def test_smoke_gate_keeps_missing_run_as_warn():
    report = run_audit(ROUTE, None, gate="smoke")
    assert report["overall"] == "WARN"


def test_audit_report_paths_are_repo_relative_for_portability():
    report = run_audit(ROUTE, None, gate="smoke")
    assert report["route_config"] == "configs/routes/models/ea_deformer.yaml"
    assert not Path(report["route_config"]).is_absolute()


def test_candidate_run_gate_accepts_test_scope_and_verified_raw_source(tmp_path):
    raw = tmp_path / "sample.mat"
    raw.write_bytes(b"raw-eeg")
    checks = []
    check_run_gate_eligibility(
        {
            "run_mode": "candidate",
            "prediction_scope": "test_only",
            "score_matrix_evidence": "genuine",
            "training_epochs": 80,
            "source_training_epochs": 80,
            "training_epochs_overridden": False,
            "raw_data_sources": [{"path": str(raw), "kind": "mat", "sha256": sha256_file(raw)}],
        },
        checks,
        gate="candidate",
        route_data=TORCH_ROUTE_DATA,
    )
    rules = {check["rule_id"]: check["status"] for check in checks}
    assert rules["RUN_MODE_CANDIDATE"] == "PASS"
    assert rules["PREDICTION_SCOPE_TEST_ONLY"] == "PASS"
    assert rules["TRAINING_EPOCHS_MATCH_ROUTE"] == "PASS"
    assert rules["SCORE_MATRIX_GENUINE"] == "PASS"
    assert rules["RAW_DATA_SOURCES_VALID"] == "PASS"
    assert "FAIL" not in rules.values()


def test_candidate_run_gate_blocks_malformed_raw_sources_without_crashing():
    checks = []
    check_run_gate_eligibility(
        {
            "run_mode": "candidate",
            "prediction_scope": "test_only",
            "score_matrix_evidence": "genuine",
            "training_epochs": 80,
            "source_training_epochs": 80,
            "training_epochs_overridden": False,
            "raw_data_sources": ["not-a-mapping"],
        },
        checks,
        gate="candidate",
        route_data=TORCH_ROUTE_DATA,
    )
    rules = {check["rule_id"]: check["status"] for check in checks}
    assert rules["RAW_DATA_SOURCES_FORMAT"] == "FAIL"


def test_candidate_run_gate_rejects_non_hex_raw_sha(tmp_path):
    raw = tmp_path / "sample.mat"
    raw.write_bytes(b"raw-eeg")
    checks = []
    check_run_gate_eligibility(
        {
            "run_mode": "candidate",
            "prediction_scope": "test_only",
            "score_matrix_evidence": "genuine",
            "training_epochs": 80,
            "source_training_epochs": 80,
            "training_epochs_overridden": False,
            "raw_data_sources": [{"path": str(raw), "kind": "mat", "sha256": "z" * 64}],
        },
        checks,
        gate="candidate",
        route_data=TORCH_ROUTE_DATA,
    )
    rules = {check["rule_id"]: check["status"] for check in checks}
    assert rules["RAW_DATA_SOURCES_SHA256"] == "FAIL"


def test_candidate_run_gate_blocks_epoch_override(tmp_path):
    raw = tmp_path / "sample.mat"
    raw.write_bytes(b"raw-eeg")
    checks = []
    check_run_gate_eligibility(
        {
            "run_mode": "candidate",
            "prediction_scope": "test_only",
            "score_matrix_evidence": "genuine",
            "training_epochs": 1,
            "source_training_epochs": 80,
            "training_epochs_overridden": True,
            "raw_data_sources": [{"path": str(raw), "kind": "mat", "sha256": sha256_file(raw)}],
        },
        checks,
        gate="candidate",
        route_data=TORCH_ROUTE_DATA,
    )
    rules = {check["rule_id"]: check["status"] for check in checks}
    assert rules["TRAINING_EPOCH_OVERRIDE"] == "FAIL"


def test_torch_classifier_candidate_requires_current_run_gate_metadata():
    checks = []
    check_run_gate_eligibility({}, checks, gate="candidate", route_data=TORCH_ROUTE_DATA)
    rules = {check["rule_id"]: check["status"] for check in checks}
    assert rules["RUN_GATE_METADATA_REQUIRED"] == "FAIL"


def test_torch_classifier_trainer_candidate_requires_current_run_gate_metadata():
    checks = []
    check_run_gate_eligibility({}, checks, gate="candidate", route_data={"training": {"trainer": "torch_classifier"}})
    rules = {check["rule_id"]: check["status"] for check in checks}
    assert rules["RUN_GATE_METADATA_REQUIRED"] == "FAIL"


def test_prediction_scope_must_match_split_subjects(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "split_manifest.yaml").write_text(
        "\n".join(
            [
                "split_id: s1",
                "train_subjects: [S1]",
                "val_subjects: [S2]",
                "test_subjects: [S3]",
                "trial_rows:",
                "  - subject_id: S1",
                "    original_trial_id: S1::t1",
                "    split: train",
                "  - subject_id: S2",
                "    original_trial_id: S2::t1",
                "    split: val",
                "  - subject_id: S3",
                "    original_trial_id: S3::t1",
                "    split: test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checks = []
    fields = {"subject_id"}
    check_prediction_scope_against_split(
        [{"subject_id": "S2"}],
        fields,
        prediction_schema(fields),
        {"prediction_scope": "test_only", "split_manifest_path": "split_manifest.yaml"},
        checks,
        run_dir=run_dir,
        gate="candidate",
    )
    rules = {check["rule_id"]: check["status"] for check in checks}
    assert rules["PREDICTION_SCOPE_SPLIT_MATCH"] == "FAIL"


def test_prediction_scope_must_match_split_trial_keys(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "split_manifest.yaml").write_text(
        "\n".join(
            [
                "split_id: s1",
                "train_subjects: [S1]",
                "val_subjects: [S2]",
                "test_subjects: [S3]",
                "trial_rows:",
                "  - subject_id: S3",
                "    original_trial_id: S3::expected_trial",
                "    split: test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checks = []
    fields = {"subject_id", "trial_id"}
    check_prediction_scope_against_split(
        [{"subject_id": "S3", "trial_id": "wrong_trial"}],
        fields,
        prediction_schema(fields),
        {
            "prediction_scope": "test_only",
            "split_manifest_path": "split_manifest.yaml",
            "evaluation_subjects": ["S3"],
        },
        checks,
        run_dir=run_dir,
        gate="candidate",
    )
    rules = {check["rule_id"]: check["status"] for check in checks}
    assert rules["PREDICTION_SCOPE_SPLIT_MATCH"] == "PASS"
    assert rules["PREDICTION_SCOPE_TRIAL_MATCH"] == "FAIL"


def test_raw_data_sources_must_match_dataset_evidence(tmp_path):
    raw_a = tmp_path / "a.mat"
    raw_b = tmp_path / "b.mat"
    raw_a.write_bytes(b"a")
    raw_b.write_bytes(b"b")
    checks = []
    add_raw_data_source_dataset_match(
        checks,
        {
            "run_mode": "candidate",
            "raw_data_sources": [{"path": str(raw_a), "kind": "mat", "sha256": sha256_file(raw_a), "exists": True}],
        },
        {
            "raw_data_sources": [
                {"path": str(raw_a), "kind": "mat", "sha256": sha256_file(raw_a), "exists": True},
                {"path": str(raw_b), "kind": "mat", "sha256": sha256_file(raw_b), "exists": True},
            ]
        },
        gate="candidate",
    )
    rules = {check["rule_id"]: check["status"] for check in checks}
    assert rules["RAW_DATA_SOURCES_DATASET_MATCH"] == "FAIL"


def test_raw_data_sources_kind_must_match_dataset_evidence(tmp_path):
    raw = tmp_path / "a.mat"
    raw.write_bytes(b"a")
    checks = []
    add_raw_data_source_dataset_match(
        checks,
        {
            "run_mode": "candidate",
            "raw_data_sources": [{"path": str(raw), "kind": "wrong", "sha256": sha256_file(raw), "exists": True}],
        },
        {
            "raw_data_sources": [{"path": str(raw), "kind": "mat", "sha256": sha256_file(raw), "exists": True}]
        },
        gate="candidate",
    )
    rules = {check["rule_id"]: check["status"] for check in checks}
    assert rules["RAW_DATA_SOURCES_DATASET_MATCH"] == "FAIL"


def test_route_model_kwargs_passthrough_requires_manifest_provenance():
    checks = []
    check_route_model_kwargs_passthrough(
        {"model_kwargs": {"drop_prob": 0.5}},
        {"model": {"name": "shallow_conv_net", "drop_prob": 0.25}},
        checks,
        gate="candidate",
    )

    rules = {check["rule_id"]: check["status"] for check in checks}
    assert rules["ROUTE_MODEL_KWARGS_PASSTHROUGH"] == "FAIL"


def test_evidence_lineage_requires_config_sha_to_match_worktree():
    route_data = yaml.safe_load(ROUTE.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    checks = []

    check_evidence_lineage(
        {
            "audit_schema_version": 2,
            "git_commit": head,
            "config_sha256": "0" * 64,
        },
        ROUTE,
        route_data,
        checks,
        gate="candidate",
    )

    rules = {check["rule_id"]: check["status"] for check in checks}
    assert rules["EVIDENCE_CONFIG_SHA_MATCHES_WORKTREE"] == "FAIL"
    assert rules["EVIDENCE_COMMIT_CONTAINS_ROUTE_AND_IMPLEMENTATION"] == "PASS"


def test_evidence_lineage_requires_commit_to_contain_route():
    route_data = yaml.safe_load(ROUTE.read_text(encoding="utf-8"))
    checks = []

    check_evidence_lineage(
        {
            "audit_schema_version": 2,
            "git_commit": "f" * 40,
            "config_sha256": sha256_file(ROUTE),
        },
        ROUTE,
        route_data,
        checks,
        gate="candidate",
    )

    rules = {check["rule_id"]: check["status"] for check in checks}
    assert rules["EVIDENCE_CONFIG_SHA_MATCHES_WORKTREE"] == "PASS"
    assert rules["EVIDENCE_COMMIT_CONTAINS_ROUTE_AND_IMPLEMENTATION"] == "FAIL"


def test_genuine_score_matrix_requires_crop_provenance(tmp_path):
    matrix = tmp_path / "score_matrix.csv"
    fieldnames = ["subject_id", "trial_id", "y_true", "crop_0", "crop_1", "crop_2", "crop_3", "crop_4"]
    with matrix.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(8):
            writer.writerow(
                {
                    "subject_id": "S1",
                    "trial_id": f"t{idx}",
                    "y_true": 1 if idx < 4 else 0,
                    "crop_0": "0.9",
                    "crop_1": "0.8",
                    "crop_2": "0.7",
                    "crop_3": "0.2",
                    "crop_4": "0.1",
                }
            )
    checks = []
    try:
        recompute_exact_metric_from_matrix(
            matrix,
            metric_name="exact_single_crop_expected_BA",
            manifest={"score_matrix_evidence": "genuine", "metric_group_keys": ["subject_id"]},
            checks=checks,
        )
    except ValueError:
        pass
    rules = {check["rule_id"]: check["status"] for check in checks}
    assert rules["SCORE_MATRIX_CROP_PROVENANCE"] == "FAIL"


def test_exact_metric_rejects_score_matrix_with_extra_crop_column(tmp_path):
    matrix = tmp_path / "score_matrix.csv"
    fieldnames = ["subject_id", "trial_id", "y_true", "crop_0", "crop_1", "crop_2", "crop_3", "crop_4", "crop_5"]
    with matrix.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(8):
            writer.writerow(
                {
                    "subject_id": "S1",
                    "trial_id": f"t{idx}",
                    "y_true": 1 if idx < 4 else 0,
                    "crop_0": "0.9",
                    "crop_1": "0.8",
                    "crop_2": "0.7",
                    "crop_3": "0.2",
                    "crop_4": "0.1",
                    "crop_5": "0.0",
                }
            )
    checks = []

    try:
        recompute_exact_metric_from_matrix(
            matrix,
            metric_name="exact_single_crop_expected_BA",
            manifest={"metric_group_keys": ["subject_id"]},
            checks=checks,
        )
    except ValueError:
        pass

    rules = {check["rule_id"]: check["status"] for check in checks}
    assert rules["METRIC_SCORE_MATRIX_SHAPE_COMPATIBLE"] == "FAIL"


def test_score_matrix_scope_must_match_split_trial_keys(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "split_manifest.yaml").write_text(
        "\n".join(
            [
                "split_id: s1",
                "train_subjects: [S1]",
                "val_subjects: [S2]",
                "test_subjects: [S3]",
                "trial_rows:",
                "  - subject_id: S3",
                "    original_trial_id: S3::test0",
                "    split: test",
                "  - subject_id: S2",
                "    original_trial_id: S2::val0",
                "    split: val",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    matrix = run_dir / "score_matrix.csv"
    fieldnames = ["subject_id", "trial_id", "y_true", "crop_0", "crop_1", "crop_2", "crop_3", "crop_4"]
    with matrix.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(8):
            writer.writerow(
                {
                    "subject_id": "S2",
                    "trial_id": f"val{idx}",
                    "y_true": 1 if idx < 4 else 0,
                    "crop_0": "0.9",
                    "crop_1": "0.8",
                    "crop_2": "0.7",
                    "crop_3": "0.2",
                    "crop_4": "0.1",
                }
            )
    checks = []
    try:
        recompute_exact_metric_from_matrix(
            matrix,
            metric_name="exact_single_crop_expected_BA",
            manifest={
                "prediction_scope": "test_only",
                "split_manifest_path": "split_manifest.yaml",
                "metric_group_keys": ["subject_id"],
            },
            checks=checks,
            run_dir=run_dir,
        )
    except ValueError:
        pass
    rules = {check["rule_id"]: check["status"] for check in checks}
    assert rules["SCORE_MATRIX_SCOPE_SPLIT_MATCH"] == "FAIL"


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


def test_candidate_gate_blocks_missing_reproducibility_metadata(tmp_path):
    route = tmp_path / "repro_route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: repro_route",
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
                "  - subject_id: s2",
                "    original_trial_id: t2",
                "    split: val",
                "  - subject_id: s3",
                "    original_trial_id: t3",
                "    split: test",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "audit_schema_version": 1,
        "route_id": "repro_route",
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
    assert rules["RUN_REPRODUCIBILITY_LOCKED"] == "FAIL"


def test_no_top4_route_skips_top4_semantics_even_when_empty_top4_column_exists(tmp_path):
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
        "subject_id,trial_id,y_score,y_pred,y_true,pred_top4\n"
        "s1,t1,0.1,0,0,\n"
        "s1,t2,0.9,1,1,\n",
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
    score_checks = [check for check in report["checks"] if check["rule_id"] == "PREDICTION_SCORE_COLUMN"]
    assert score_checks[-1]["status"] == "PASS"


def test_no_top4_route_does_not_require_score_column_for_candidate(tmp_path):
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
    prediction.write_text("subject_id,trial_id,y_pred,y_true\ns1,t1,0,0\ns1,t2,1,1\n", encoding="utf-8")
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
        "primary_metric": "no_top4_BA",
        "metrics": {"no_top4_BA": 1.0},
        "prediction_csv": "predictions.csv",
        "prediction_sha256": sha256_file(prediction),
        "prediction_record_level": "trial",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(route, run_dir, gate="candidate")
    rules = {check["rule_id"]: check["status"] for check in report["checks"]}
    assert rules["PREDICTION_SCORE_COLUMN"] == "PASS"
    assert rules["METRIC_RECOMPUTE_BA"] == "PASS"
    assert rules["PRIMARY_METRIC_RECOMPUTE"] == "PASS"


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


def test_exact_metric_with_score_matrix_does_not_require_y_pred(tmp_path):
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
    lines = ["subject_id,trial_id,y_score,y_true,pred_top4"]
    for idx in range(8):
        label = 1 if idx < 4 else 0
        score = 1.0 - idx * 0.01
        lines.append(f"s1,t{idx},{score},{label},{label}")
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
    assert rules["METRIC_RECOMPUTE_BA"] == "PASS"
    assert rules["PRIMARY_METRIC_RECOMPUTE"] == "PASS"


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
                "label_scope:",
                "  train: available",
                "  val: available",
                "  test: available_for_audit_only",
                "  pseudo_public: hidden_until_audit",
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
                "trial_rows:",
                "  - subject_id: s1",
                "    original_trial_id: t1",
                "    split: train",
                "  - subject_id: s2",
                "    original_trial_id: t2",
                "    split: val",
                "  - subject_id: s3",
                "    original_trial_id: t3",
                "    split: test",
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
        **reproducibility_metadata(),
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
    assert checks["PREDICTION_TOP4_TRUTH_BALANCE"]["status"] == "WARN"
    assert "not checked" in checks["PREDICTION_TOP4_TRUTH_BALANCE"]["message"]

    candidate_report = run_audit(route, run_dir, gate="candidate")
    candidate_checks = {check["rule_id"]: check for check in candidate_report["checks"]}
    assert candidate_checks["PREDICTION_TOP4_TRUTH_BALANCE"]["status"] == "FAIL"


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


def test_folded_trial_rows_allow_cross_fold_original_trial_reuse():
    subject_errors, trial_errors = split_leakage_errors(
        {
            "trial_rows": [
                {"fold": 0, "subject_id": "s1", "original_trial_id": "t1", "split": "train"},
                {"fold": 0, "subject_id": "s2", "original_trial_id": "t2", "split": "test"},
                {"fold": 1, "subject_id": "s1", "original_trial_id": "t1", "split": "test"},
                {"fold": 1, "subject_id": "s2", "original_trial_id": "t2", "split": "train"},
            ]
        }
    )
    assert not subject_errors
    assert not trial_errors


def test_folded_trial_rows_still_reject_within_fold_original_trial_leakage():
    _, trial_errors = split_leakage_errors(
        {
            "trial_rows": [
                {"fold": 0, "subject_id": "s1", "original_trial_id": "t1", "split": "train"},
                {"fold": 0, "subject_id": "s1", "original_trial_id": "t1", "split": "test"},
            ]
        }
    )
    assert trial_errors


def test_split_evidence_consistency_requires_subject_lists_to_match_trial_rows():
    errors = split_evidence_consistency_errors(
        {
            "train_subjects": ["s1", "s2"],
            "val_subjects": ["s3"],
            "test_subjects": ["s4"],
            "trial_rows": [
                {"subject_id": "s1", "original_trial_id": "t1", "split": "train"},
                {"subject_id": "s3", "original_trial_id": "t2", "split": "val"},
                {"subject_id": "s4", "original_trial_id": "t3", "split": "holdout"},
            ],
        }
    )
    assert any("train_subjects" in item for item in errors)
    assert any("split must be train/val/test" in item for item in errors)


def test_split_evidence_consistency_rejects_unconsumed_fold_rows():
    errors = split_evidence_consistency_errors(
        {
            "folds": [
                {"fold": 0, "train_subjects": ["s1"], "val_subjects": ["s2"], "test_subjects": ["s3"]},
            ],
            "trial_rows": [
                {"fold": 0, "subject_id": "s1", "original_trial_id": "t1", "split": "train"},
                {"fold": 0, "subject_id": "s2", "original_trial_id": "t2", "split": "val"},
                {"fold": 0, "subject_id": "s3", "original_trial_id": "t3", "split": "test"},
                {"fold": 999, "subject_id": "s9", "original_trial_id": "t9", "split": "test"},
            ],
        }
    )
    assert any("fold is not declared" in item for item in errors)
    assert any("not covered" in item for item in errors)


def test_reproducibility_check_rejects_mismatched_seed_and_nondeterminism():
    manifest = {
        "seed": 42,
        **reproducibility_metadata(),
    }
    manifest["determinism"] = dict(manifest["determinism"])
    manifest["determinism"]["python_seed"] = 7
    manifest["determinism"]["deterministic_algorithms"] = False
    manifest["determinism"]["cudnn_benchmark"] = True
    manifest["determinism"]["batch_order"] = {"policy": "unseeded_shuffle", "sampler_seed": 7}
    manifest["checkpoint_selection"] = dict(manifest["checkpoint_selection"])
    manifest["checkpoint_selection"]["tie_break"] = "latest_epoch"
    manifest["crop_policy"] = {"name": "random", "selection": "per_trial_uniform_crop", "random_seed": 7, "rng": "numpy.default_rng", "tie_break": "stable_input_order"}

    checks = []
    check_reproducibility_manifest(manifest, checks, gate="candidate", root=Path.cwd())

    check = checks[-1]
    assert check["rule_id"] == "RUN_REPRODUCIBILITY_LOCKED"
    assert check["status"] == "FAIL"
    assert "python_seed" in check["message"]
    assert "deterministic_algorithms" in check["message"]
    assert "crop_policy.random_seed" in check["message"]


def test_candidate_gate_rejects_unconsumed_fold_rows_end_to_end(tmp_path):
    route = tmp_path / "bad_fold_rows_route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: bad_fold_rows_route",
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
                "folds:",
                "  - fold: 0",
                "    train_subjects: [s1]",
                "    val_subjects: [s2]",
                "    test_subjects: [s3]",
                "trial_rows:",
                "  - fold: 0",
                "    subject_id: s1",
                "    original_trial_id: t1",
                "    split: train",
                "  - fold: 0",
                "    subject_id: s2",
                "    original_trial_id: t2",
                "    split: val",
                "  - fold: 0",
                "    subject_id: s3",
                "    original_trial_id: t3",
                "    split: test",
                "  - fold: 999",
                "    subject_id: s9",
                "    original_trial_id: t9",
                "    split: test",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "audit_schema_version": 1,
        "route_id": "bad_fold_rows_route",
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
        **reproducibility_metadata(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(route, run_dir, gate="candidate")
    rules = {check["rule_id"]: check["status"] for check in report["checks"]}
    assert rules["RUN_SPLIT_EVIDENCE_VALID"] == "FAIL"


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


def test_dataset_label_scope_rejects_publicly_available_test_labels():
    errors = dataset_label_scope_errors(
        {
            "label_scope": {
                "train": "available",
                "val": "available",
                "test": "available",
                "pseudo_public": "available",
            }
        }
    )

    assert any("label_scope.test" in error for error in errors)
    assert any("label_scope.pseudo_public" in error for error in errors)


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


def test_promotion_audit_uses_strict_key_value_parser(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.audit_experiment.ROOT", tmp_path)
    promotion = tmp_path / "reports" / "promotion_audits" / "summary_route_promotion.md"
    promotion.parent.mkdir(parents=True)
    promotion.write_text(
        "\n".join(
            [
                "-route_id: summary_route",
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

    rule = next(check for check in checks if check["rule_id"] == "PROMOTION_AUDIT_FORMAT")
    assert rule["status"] == "FAIL"
    assert "route_id" in rule["message"]


def test_promotion_audit_requires_top4_rules_from_route_policy(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.audit_experiment.ROOT", tmp_path)
    critical_rules = [
        "MANIFEST_VALID",
        "PRIMARY_METRIC_RECOMPUTE",
        "PRIMARY_METRIC_REPORTED",
        "RUN_DATASET_EVIDENCE_VALID",
        "RUN_SPLIT_EVIDENCE_VALID",
        "RUN_SPLIT_EVIDENCE_CONSISTENT",
        "RUN_REPRODUCIBILITY_LOCKED",
    ]
    audit = tmp_path / "reports" / "audits" / "candidate.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(
        json.dumps({"route_id": "summary_route", "gate": "candidate", "overall": "PASS", "checks": [{"rule_id": rule, "status": "PASS"} for rule in critical_rules]}),
        encoding="utf-8",
    )
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
    check_promotion_audit("summary_route", promotion, checks, top4_required=True)

    rule = next(check for check in checks if check["rule_id"] == "PROMOTION_AUDIT_CANDIDATE_RULES")
    assert rule["status"] == "FAIL"
    assert "PREDICTION_TOP4_RANKING" in rule["message"]


def test_split_evidence_allows_empty_validation_holdout():
    split = {
        "split_id": "holdout",
        "subject_group_split": True,
        "status": "ready",
        "train_subjects": ["s1", "s2"],
        "val_subjects": [],
        "test_subjects": ["s3"],
        "trial_rows": [
            {"subject_id": "s1", "original_trial_id": "t1", "split": "train"},
            {"subject_id": "s2", "original_trial_id": "t2", "split": "train"},
            {"subject_id": "s3", "original_trial_id": "t3", "split": "test"},
        ],
    }

    assert split_manifest_has_formal_evidence(split)
    assert split_evidence_consistency_errors(split) == []


def test_split_evidence_accepts_trial_rows_only_membership():
    split = {
        "split_id": "rows_only",
        "subject_group_split": True,
        "status": "ready",
        "trial_rows": [
            {"subject_id": "s1", "original_trial_id": "t1", "split": "train"},
            {"subject_id": "s2", "original_trial_id": "t2", "split": "val"},
            {"subject_id": "s3", "original_trial_id": "t3", "split": "test"},
        ],
    }

    assert split_manifest_has_formal_evidence(split)
    assert split_evidence_consistency_errors(split) == []


def test_split_evidence_accepts_folded_trial_rows_only_membership():
    split = {
        "split_id": "rows_only_folded",
        "subject_group_split": True,
        "status": "ready",
        "trial_rows": [
            {"fold": 0, "subject_id": "s1", "original_trial_id": "f0_t1", "split": "train"},
            {"fold": 0, "subject_id": "s2", "original_trial_id": "f0_t2", "split": "test"},
            {"fold": 1, "subject_id": "s2", "original_trial_id": "f1_t1", "split": "train"},
            {"fold": 1, "subject_id": "s1", "original_trial_id": "f1_t2", "split": "test"},
        ],
    }

    assert split_manifest_has_formal_evidence(split)
    assert split_evidence_consistency_errors(split) == []


def test_split_evidence_rejects_trial_rows_only_without_test_membership():
    split = {
        "split_id": "train_only_rows",
        "subject_group_split": True,
        "status": "ready",
        "trial_rows": [
            {"subject_id": "s1", "original_trial_id": "t1", "split": "train"},
            {"subject_id": "s2", "original_trial_id": "t2", "split": "train"},
        ],
    }

    assert not split_manifest_has_formal_evidence(split)
    assert "trial_rows must derive non-empty train and test subject membership" in split_evidence_consistency_errors(split)


def test_split_evidence_rejects_folded_trial_rows_without_test_membership():
    split = {
        "split_id": "folded_train_only_rows",
        "subject_group_split": True,
        "status": "ready",
        "trial_rows": [
            {"fold": 0, "subject_id": "s1", "original_trial_id": "t1", "split": "train"},
            {"fold": 0, "subject_id": "s2", "original_trial_id": "t2", "split": "train"},
        ],
    }

    assert not split_manifest_has_formal_evidence(split)
    assert "trial_rows fold 0 must derive non-empty train and test subject membership" in split_evidence_consistency_errors(split)


def test_split_evidence_rejects_mixed_folded_and_unfolded_trial_rows():
    split = {
        "split_id": "mixed_rows",
        "subject_group_split": True,
        "status": "ready",
        "trial_rows": [
            {"fold": 0, "subject_id": "s1", "original_trial_id": "t1", "split": "train"},
            {"subject_id": "s1", "original_trial_id": "t1", "split": "test"},
        ],
    }

    assert not split_manifest_has_formal_evidence(split)
    assert "trial_rows must either all declare fold or none declare fold" in split_evidence_consistency_errors(split)
    subject_errors, trial_errors = split_leakage_errors(split)
    assert any("mix folded and unfolded" in error for error in subject_errors)
    assert any("mix folded and unfolded" in error for error in trial_errors)


def test_primary_metric_recompute_uses_subject_mean_for_prediction_ba(tmp_path):
    prediction_csv = tmp_path / "predictions.csv"
    lines = ["subject_id,trial_id,y_true,y_pred,pred_top4"]
    lines.extend(["s1,s1_t0,0,0,0", "s1,s1_t1,1,1,1"])
    for idx in range(5):
        lines.append(f"s2,s2_n{idx},0,0,0")
    for idx in range(5):
        lines.append(f"s2,s2_p{idx},1,0,0")
    prediction_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = build_metric_report(route_id="r1", prediction_csv=prediction_csv, primary_metric="no_top4_BA")
    with prediction_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = set(rows[0])
    recomputed = recompute_primary_metric_from_predictions(
        metric_name="no_top4_BA",
        rows=rows,
        fields=fields,
        schema=prediction_schema(fields),
        manifest={"metric_group_keys": ["subject_id"]},
    )

    assert report["metrics"]["no_top4_BA"] == 0.75
    assert recomputed == report["metrics"]["no_top4_BA"]


def test_primary_metric_report_rejects_nan_metric_value():
    checks = []
    compare_metric(checks, metric_name="no_top4_BA", recomputed=0.75, manifest={"metrics": {"no_top4_BA": float("nan")}})

    assert checks[-1]["rule_id"] == "PRIMARY_METRIC_REPORTED"
    assert checks[-1]["status"] == "FAIL"
