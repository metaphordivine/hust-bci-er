import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

import hust_bci_er.training.reproducibility as reproducibility_module
from hust_bci_er.audit.manifest import sha256_file, validate_manifest
from hust_bci_er.audit.run_manifest import write_run_manifest
from hust_bci_er.evaluation.crop_policy import crop_policy_manifest, select_crop_matrix, worst_crop_score
from hust_bci_er.evaluation.protocols.runner import build_protocol_jobs, materialize_protocol_run
from hust_bci_er.training.reproducibility import dataloader_worker_seed
from scripts.audit_experiment import run_audit


ROUTE = Path("configs/routes/models/ea_deformer.yaml")
ROUTE2 = Path("configs/routes/models/sliding_window_eegnet.yaml")


def lock_pythonhashseed(monkeypatch, seed: int) -> None:
    value = str(seed)
    monkeypatch.setenv("PYTHONHASHSEED", value)
    monkeypatch.setattr(reproducibility_module, "PROCESS_START_PYTHONHASHSEED", value)


def run_reproducibility_subprocess(*, seed: int, pythonhashseed: str | None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if pythonhashseed is None:
        env.pop("PYTHONHASHSEED", None)
    else:
        env["PYTHONHASHSEED"] = pythonhashseed
    code = (
        "import json; "
        "from hust_bci_er.training.reproducibility import ReproducibilityConfig, apply_reproducibility; "
        f"print(json.dumps(apply_reproducibility(ReproducibilityConfig(seed={seed})), sort_keys=True))"
    )
    return subprocess.run([sys.executable, "-c", code], cwd=Path.cwd(), env=env, text=True, capture_output=True, timeout=60)


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


def test_protocol_runner_records_multiple_param_indices_in_shared_p3_split(tmp_path):
    manifest = materialize_protocol_run(
        "p3",
        [ROUTE],
        run_dir=tmp_path / "p3_run",
        outer_folds=2,
        inner_folds=2,
        grid_sizes={"ea_deformer": 3},
    )
    inner_job = next(job for job in manifest["jobs"] if job["stage"] == "inner_select")

    split_payload = yaml.safe_load((tmp_path / "p3_run" / inner_job["split_manifest_path"]).read_text(encoding="utf-8"))

    assert "param_index" not in split_payload
    assert split_payload["param_indices"] == [0, 1, 2]
    assert "seed" not in split_payload
    assert split_payload["seeds"] == [123, 124, 125]


def test_protocol_runner_records_multiple_routes_in_shared_split_contract(tmp_path):
    manifest = materialize_protocol_run("p2", [ROUTE, ROUTE2], run_dir=tmp_path / "p2_multi_route")
    first_job = manifest["jobs"][0]

    split_payload = yaml.safe_load((tmp_path / "p2_multi_route" / first_job["split_manifest_path"]).read_text(encoding="utf-8"))

    assert "route_id" not in split_payload
    assert split_payload["route_ids"] == ["ea_deformer", "sliding_window_eegnet"]
    assert any(job_id.startswith("p2__ea_deformer__") for job_id in split_payload["job_ids"])
    assert any(job_id.startswith("p2__sliding_window_eegnet__") for job_id in split_payload["job_ids"])


def test_write_run_manifest_locks_artifact_hashes(monkeypatch, tmp_path):
    lock_pythonhashseed(monkeypatch, 42)
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


def test_write_run_manifest_requires_locked_pythonhashseed_for_default_determinism(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.setattr(reproducibility_module, "PROCESS_START_PYTHONHASHSEED", None)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    predictions = run_dir / "predictions.csv"
    predictions.write_text(
        "subject_id,trial_id,y_score,y_pred,y_true,pred_top4\n"
        + "\n".join(f"s1,t{idx},{1.0 - idx * 0.01},{1 if idx < 4 else 0},{1 if idx < 4 else 0},{1 if idx < 4 else 0}" for idx in range(8))
        + "\n",
        encoding="utf-8",
    )

    try:
        write_run_manifest(
            route_config_path=ROUTE,
            run_dir=run_dir,
            prediction_csv=predictions,
            metrics={"exact_single_crop_expected_BA": 1.0},
            command="test",
        )
    except ValueError as exc:
        assert "restart Python with PYTHONHASHSEED=42" in str(exc)
    else:
        raise AssertionError("default run manifest determinism should require a locked PYTHONHASHSEED")


def test_write_run_manifest_reports_missing_route_fields(monkeypatch, tmp_path):
    lock_pythonhashseed(monkeypatch, 42)
    root = tmp_path
    route = root / "configs" / "routes" / "models" / "bad.yaml"
    route.parent.mkdir(parents=True)
    route.write_text(
        "\n".join(
            [
                "route_id: bad",
                "dataset_version: train_v1",
                "split_id: p1_seed42_fold0",
                "seed: 42",
                "",
            ]
        ),
        encoding="utf-8",
    )
    run_dir = root / "outputs" / "bad" / "run"
    run_dir.mkdir(parents=True)
    predictions = run_dir / "predictions.csv"
    predictions.write_text("subject_id,trial_id,y_true,y_pred\ns1,t1,1,1\n", encoding="utf-8")

    try:
        write_run_manifest(
            route_config_path=route,
            run_dir=run_dir,
            prediction_csv=predictions,
            metrics={"no_top4_BA": 1.0},
            command="test",
        )
    except ValueError as exc:
        assert "route config missing required field: evaluation" in str(exc)
    else:
        raise AssertionError("write_run_manifest should reject route configs missing evaluation")


def test_write_run_manifest_accepts_job_specific_split_and_seed(monkeypatch, tmp_path):
    protocol_run = tmp_path / "protocol"
    protocol_manifest = materialize_protocol_run("p1", [ROUTE], run_dir=protocol_run, seeds=[123], n_folds=2)
    job = protocol_manifest["jobs"][0]
    lock_pythonhashseed(monkeypatch, int(job["seed"]))

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


def test_write_run_manifest_requires_split_evidence_for_split_override(monkeypatch, tmp_path):
    lock_pythonhashseed(monkeypatch, 42)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    predictions = run_dir / "predictions.csv"
    predictions.write_text(
        "subject_id,trial_id,y_score,y_pred,y_true,pred_top4\n"
        + "\n".join(f"s1,t{idx},{1.0 - idx * 0.01},{1 if idx < 4 else 0},{1 if idx < 4 else 0},{1 if idx < 4 else 0}" for idx in range(8))
        + "\n",
        encoding="utf-8",
    )

    try:
        write_run_manifest(
            route_config_path=ROUTE,
            run_dir=run_dir,
            prediction_csv=predictions,
            split_id="p1_seed42_fold0__derived",
            metrics={"exact_single_crop_expected_BA": 1.0},
            command="test",
        )
    except ValueError as exc:
        assert "split_manifest_path is required" in str(exc)
    else:
        raise AssertionError("write_run_manifest should reject split_id override without split evidence")


def test_write_run_manifest_records_group_keys_and_route_crop_tie_break(monkeypatch, tmp_path):
    lock_pythonhashseed(monkeypatch, 42)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    predictions = run_dir / "predictions.csv"
    predictions.write_text(
        "seed,fold,subject_id,trial_id,y_score,y_pred,y_true,pred_top4\n"
        + "\n".join(f"42,0,s1,t{idx},{1.0 - idx * 0.01},{1 if idx < 4 else 0},{1 if idx < 4 else 0},{1 if idx < 4 else 0}" for idx in range(8))
        + "\n",
        encoding="utf-8",
    )

    manifest = write_run_manifest(
        route_config_path=Path("configs/routes/models/sliding_window_eegnet.yaml"),
        run_dir=run_dir,
        prediction_csv=predictions,
        metrics={"exact_single_crop_expected_BA": 1.0},
        command="test",
        top4_group_keys=("seed", "fold", "subject_id"),
        metric_group_keys=("seed", "fold", "subject_id"),
    )

    assert manifest["top4_group_keys"] == ["seed", "fold", "subject_id"]
    assert manifest["metric_group_keys"] == ["seed", "fold", "subject_id"]
    assert manifest["crop_policy"]["name"] == "sliding_window_vote"
    assert manifest["crop_policy"]["tie_break"] == "mean_score"


def test_reproducibility_helpers_define_worker_seed_and_apply():
    proc = run_reproducibility_subprocess(seed=123, pythonhashseed="123")

    assert proc.returncode == 0, proc.stderr
    applied = json.loads(proc.stdout)
    assert applied["python_seed"] == 123
    assert applied["python_hash_seed"] == 123
    assert applied["pythonhashseed_env"] == "123"
    assert applied["numpy_seed"] == 123
    assert applied["torch_seed"] == 123
    assert applied["dataloader_worker_seed_base"] == 123
    assert dataloader_worker_seed(500, 3) == 503


def test_reproducibility_rejects_mismatched_pythonhashseed():
    proc = run_reproducibility_subprocess(seed=123, pythonhashseed="999")

    assert proc.returncode != 0
    assert "PYTHONHASHSEED mismatch" in proc.stderr


def test_reproducibility_rejects_missing_pythonhashseed():
    proc = run_reproducibility_subprocess(seed=123, pythonhashseed=None)

    assert proc.returncode != 0
    assert "restart Python with PYTHONHASHSEED=123" in proc.stderr


def test_crop_policy_random_and_worst_are_deterministic():
    assert crop_policy_manifest("random", seed=7)["random_seed"] == 7
    assert crop_policy_manifest("worst", seed=7)["tie_break"] == "lowest_assignment_index"
    assert crop_policy_manifest("sliding_window_vote", seed=7)["tie_break"] == "mean_score"
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
