import json
from pathlib import Path

import yaml

from hust_bci_er.audit.manifest import sha256_file, validate_manifest
from hust_bci_er.audit.run_manifest import write_run_manifest
from hust_bci_er.evaluation.crop_policy import crop_policy_manifest, select_crop_matrix, worst_crop_score
from hust_bci_er.evaluation.protocols.runner import build_protocol_jobs, materialize_protocol_run
from hust_bci_er.training.reproducibility import ReproducibilityConfig, apply_reproducibility, dataloader_worker_seed
from scripts.audit_experiment import run_audit


ROUTE = Path("configs/routes/models/ea_deformer.yaml")


def test_protocol_runner_materializes_p2_crop_jobs(tmp_path):
    manifest = materialize_protocol_run("p2", [ROUTE], run_dir=tmp_path / "p2_run")
    assert manifest["protocol"] == "p2_pseudo_public_holdout"
    assert len(manifest["jobs"]) == 8
    policies = [job["crop_policy"]["name"] for job in manifest["jobs"] if job["crop_policy"]]
    assert policies == ["crop1", "crop2", "crop3", "crop4", "crop5", "random", "worst"]
    random_job = next(job for job in manifest["jobs"] if job["crop_policy"] and job["crop_policy"]["name"] == "random")
    worst_job = next(job for job in manifest["jobs"] if job["crop_policy"] and job["crop_policy"]["name"] == "worst")
    assert random_job["crop_policy"]["random_seed"] == 42
    assert worst_job["crop_policy"]["selection"] == "label_aware_min_metric_stress_test"
    assert worst_job["crop_policy"]["tie_break"] == "lowest_assignment_index"
    assert random_job["source_split_id"] == "p1_seed42_fold0"
    assert random_job["split_id"] == "p1_seed42_fold0__p2_holdout999_train42"
    assert random_job["split_manifest_path"] == "splits/p1_seed42_fold0__p2_holdout999_train42.yaml"
    assert len(random_job["split_sha256"]) == 64
    assert manifest["route_locks"][0]["source_split_manifest_path"] == "configs/splits/p1_seed42_fold0.yaml"
    assert len(manifest["experiment_gate_job_ids"]) == 7
    assert manifest["artifact_only_job_ids"] == ["p2__ea_deformer__train_seed42"]
    assert "artifact-only" in manifest["completion_rule"]
    assert (tmp_path / "p2_run" / "protocol_run_manifest.json").exists()
    split_path = tmp_path / "p2_run" / random_job["split_manifest_path"]
    assert split_path.exists()
    assert random_job["split_sha256"] == sha256_file(split_path)
    assert worst_job["split_sha256"] == sha256_file(split_path)
    split_contract = yaml.safe_load(split_path.read_text(encoding="utf-8"))
    assert len(split_contract["job_ids"]) == 8
    assert [policy["name"] for policy in split_contract["crop_policies"]] == ["crop1", "crop2", "crop3", "crop4", "crop5", "random", "worst"]


def test_protocol_runner_counts_p1_and_p3_jobs():
    _, p1_jobs = build_protocol_jobs("p1", [ROUTE], seeds=[1, 2], n_folds=3)
    assert len(p1_jobs) == 6
    assert {job.split_id for job in p1_jobs} == {
        "p1_seed42_fold0__p1_seed1_fold0",
        "p1_seed42_fold0__p1_seed1_fold1",
        "p1_seed42_fold0__p1_seed1_fold2",
        "p1_seed42_fold0__p1_seed2_fold0",
        "p1_seed42_fold0__p1_seed2_fold1",
        "p1_seed42_fold0__p1_seed2_fold2",
    }
    _, p3_jobs = build_protocol_jobs("p3", [ROUTE], outer_folds=2, inner_folds=2, grid_sizes={"ea_deformer": 3})
    assert len(p3_jobs) == 14
    assert all(job.split_id != job.source_split_id for job in p3_jobs)


def test_write_run_manifest_locks_artifact_hashes(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    predictions = run_dir / "predictions.csv"
    predictions.write_text(
        "subject_id,trial_id,y_score,y_pred,y_true,pred_top4\n"
        + "\n".join(f"s1,t{idx},{1.0 - idx * 0.01},{1 if idx < 4 else 0},{1 if idx < 4 else 0},{1 if idx < 4 else 0}" for idx in range(8))
        + "\n",
        encoding="utf-8",
    )
    score_matrix = run_dir / "score_matrix.csv"
    score_matrix.write_text(
        "subject_id,trial_id,y_true,crop_0,crop_1,crop_2,crop_3,crop_4\n"
        + "\n".join(f"s1,t{idx},{1 if idx < 4 else 0},{1.0 - idx * 0.01},{1.0 - idx * 0.01},{1.0 - idx * 0.01},{1.0 - idx * 0.01},{1.0 - idx * 0.01}" for idx in range(8))
        + "\n",
        encoding="utf-8",
    )

    manifest = write_run_manifest(
        route_config_path=ROUTE,
        run_dir=run_dir,
        prediction_csv=predictions,
        score_matrix_csv=score_matrix,
        metrics={"exact_single_crop_expected_BA": 1.0},
        command=["python", "scripts/run_evaluation_protocol.py", "--protocol", "p1"],
    )

    assert manifest["prediction_sha256"] == sha256_file(predictions)
    assert manifest["metric_inputs"]["score_matrix_sha256"] == sha256_file(score_matrix)
    assert manifest["environment_lock"]["files"]
    assert validate_manifest(run_dir / "manifest.json", root=Path.cwd(), route_data=None) == []
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["audit_schema_version"] == 2


def test_write_run_manifest_accepts_job_specific_split_and_seed(tmp_path):
    protocol_run = tmp_path / "protocol"
    protocol_manifest = materialize_protocol_run("p1", [ROUTE], run_dir=protocol_run, seeds=[123], n_folds=2)
    job = protocol_manifest["jobs"][0]

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    predictions = run_dir / "predictions.csv"
    predictions.write_text(
        "subject_id,trial_id,y_score,y_pred,y_true,pred_top4\n"
        + "\n".join(f"s1,t{idx},{1.0 - idx * 0.01},{1 if idx < 4 else 0},{1 if idx < 4 else 0},{1 if idx < 4 else 0}" for idx in range(8))
        + "\n",
        encoding="utf-8",
    )
    score_matrix = run_dir / "score_matrix.csv"
    score_matrix.write_text(
        "subject_id,trial_id,y_true,crop_0,crop_1,crop_2,crop_3,crop_4\n"
        + "\n".join(f"s1,t{idx},{1 if idx < 4 else 0},{1.0 - idx * 0.01},{1.0 - idx * 0.01},{1.0 - idx * 0.01},{1.0 - idx * 0.01},{1.0 - idx * 0.01}" for idx in range(8))
        + "\n",
        encoding="utf-8",
    )
    dataset_path = run_dir / "dataset_manifest.yaml"
    dataset_path.write_text(
        "dataset_version: train_v1\nstatus: ready\ndata_sources:\n  - path: data/train.csv\n    kind: table\nchecksum_manifest:\n  - path: data/train.csv\n    sha256: '"
        + "1" * 64
        + "'\n",
        encoding="utf-8",
    )
    split_path = run_dir / "job_split.yaml"
    split_payload = yaml.safe_load((protocol_run / job["split_manifest_path"]).read_text(encoding="utf-8"))
    split_payload.update(
        {
            "status": "ready",
            "train_subjects": ["s1"],
            "val_subjects": ["s2"],
            "test_subjects": ["s3"],
            "trial_rows": [
                {"subject_id": "s1", "original_trial_id": "t1", "split": "train"},
                {"subject_id": "s2", "original_trial_id": "t2", "split": "val"},
                {"subject_id": "s3", "original_trial_id": "t3", "split": "test"},
            ],
        }
    )
    split_path.write_text(yaml.safe_dump(split_payload, sort_keys=False), encoding="utf-8")

    manifest = write_run_manifest(
        route_config_path=ROUTE,
        run_dir=run_dir,
        prediction_csv=predictions,
        dataset_manifest_path=dataset_path,
        split_manifest_path=split_path,
        split_id=job["split_id"],
        source_split_id=job["source_split_id"],
        seed=job["seed"],
        source_seed=42,
        score_matrix_csv=score_matrix,
        metrics={"exact_single_crop_expected_BA": 1.0},
        command="test",
        crop_policy=job["crop_policy"],
    )

    assert manifest["split_id"] == job["split_id"]
    assert manifest["source_split_id"] == "p1_seed42_fold0"
    assert manifest["seed"] == 123
    assert manifest["source_seed"] == 42
    assert manifest["split_sha256"] == sha256_file(split_path)
    assert validate_manifest(run_dir / "manifest.json", root=Path.cwd(), route_data=None) == []
    report = run_audit(ROUTE, run_dir, gate="smoke")
    rules = {check["rule_id"]: check["status"] for check in report["checks"]}
    assert rules["MANIFEST_VALID"] == "PASS"
    assert rules["RUN_REPRODUCIBILITY_LOCKED"] == "PASS"
    assert rules["RUN_SPLIT_EVIDENCE_VALID"] == "PASS"
    assert rules["RUN_SPLIT_EVIDENCE_CONSISTENT"] == "PASS"
    assert rules["PRIMARY_METRIC_RECOMPUTE"] == "PASS"


def test_reproducibility_helpers_define_worker_seed_and_apply():
    applied = apply_reproducibility(ReproducibilityConfig(seed=123, dataloader_worker_seed_base=500))
    assert applied["python_seed"] == 123
    assert applied["numpy_seed"] == 123
    assert applied["torch_seed"] == 123
    assert applied["dataloader_worker_seed_base"] == 500
    assert dataloader_worker_seed(500, 3) == 503


def test_crop_policy_random_and_worst_are_deterministic():
    assert crop_policy_manifest("random", seed=7)["random_seed"] == 7
    assert crop_policy_manifest("worst", seed=7)["tie_break"] == "lowest_assignment_index"
    mat = [[0, 1, 2], [3, 4, 5]]
    assert select_crop_matrix(mat, "random", random_state=1).tolist() == select_crop_matrix(mat, "random", random_state=1).tolist()
    try:
        select_crop_matrix(mat, "random")
    except ValueError as exc:
        assert "random_state" in str(exc)
    else:
        raise AssertionError("random crop without seed should fail")
    score, idx = worst_crop_score([[0.0, 0.0], [0.0, 0.0]], [0, 1], lambda row, y: 0.5)
    assert score == 0.5
    assert idx == 0
