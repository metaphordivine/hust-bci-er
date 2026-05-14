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
        "split_id": "p1_seed42_fold0",
        "split_sha256": sha256_file(Path("configs/splits/p1_seed42_fold0.yaml")),
        "seed": 42,
        "command": "test",
        "primary_metric": "exact_single_crop_expected_BA",
        "prediction_csv": "predictions.csv",
        "prediction_sha256": sha256_file(prediction),
        "prediction_record_level": "trial",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = run_audit(route, run_dir, gate="candidate")
    top4_checks = [check for check in report["checks"] if check["rule_id"] == "PREDICTION_TOP4_GROUPS"]
    assert top4_checks[-1]["status"] == "PASS"
