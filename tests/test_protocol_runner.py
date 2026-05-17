import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

import hust_bci_er.training.reproducibility as reproducibility_module
from hust_bci_er.audit.manifest import sha256_file, validate_manifest
from hust_bci_er.audit.run_manifest import write_run_manifest
from hust_bci_er.evaluation.crop_policy import crop_policy_manifest, select_crop_matrix, worst_crop_score
from hust_bci_er.evaluation.protocols.runner import build_protocol_jobs, execute_protocol_jobs, materialize_protocol_run
from hust_bci_er.evaluation.protocols.summary import write_protocol_summary
from hust_bci_er.training.reproducibility import dataloader_worker_seed
from scripts.audit_experiment import run_audit


ROUTE = Path("configs/routes/models/ea_deformer.yaml")
ROUTE2 = Path("configs/routes/models/sliding_window_eegnet.yaml")


def write_hust_mat(path: Path, *, samples_per_trial: int = 8) -> None:
    h5py = __import__("pytest").importorskip("h5py")
    total = samples_per_trial * 4
    with h5py.File(path, "w") as f:
        f.create_dataset("EEG_data_neu", data=np.zeros((30, total), dtype=np.float32))
        f.create_dataset("EEG_data_pos", data=np.ones((30, total), dtype=np.float32))


def write_hust_data_root(root: Path, *, subjects_per_cohort: int = 6) -> Path:
    root.mkdir()
    for idx in range(subjects_per_cohort):
        write_hust_mat(root / f"DEP{idx:03d}timedata.mat")
        write_hust_mat(root / f"HC{idx:03d}timedata.mat")
    return root


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
    assert "reusable checkpoint" in manifest["completion_rule"]
    train_job = next(job for job in manifest["jobs"] if job["stage"] == "train_holdout_model")
    assert train_job["expected_artifacts"] == ["config_snapshot.yaml", "checkpoint.pt", "train_manifest.json"]
    assert random_job["reuse_checkpoint_job_id"] == "p2__ea_deformer__train_seed42"
    assert random_job["reuse_checkpoint_path"] == "job_runs/p2__ea_deformer__train_seed42/checkpoint.pt"
    assert (tmp_path / "p2_run" / "protocol_run_manifest.json").exists()
    split_path = tmp_path / "p2_run" / random_job["split_manifest_path"]
    assert split_path.exists()
    assert random_job["split_sha256"] == sha256_file(split_path)
    assert worst_job["split_sha256"] == sha256_file(split_path)
    split_contract = yaml.safe_load(split_path.read_text(encoding="utf-8"))
    assert len(split_contract["job_ids"]) == 8
    assert [policy["name"] for policy in split_contract["crop_policies"]] == ["crop1", "crop2", "crop3", "crop4", "crop5", "random", "worst"]


def test_protocol_execute_results_list_artifact_only_skips(tmp_path):
    run_dir = tmp_path / "p2_run"
    manifest = materialize_protocol_run("p2", [ROUTE], run_dir=run_dir)
    manifest["default_execute_device"] = "cuda"

    results = execute_protocol_jobs(manifest, protocol_run_manifest_path=run_dir / "protocol_run_manifest.json", max_jobs=0)

    skipped = [item for item in results if item["status"] == "SKIPPED_ARTIFACT_ONLY"]
    assert [item["job_id"] for item in skipped] == ["p2__ea_deformer__train_seed42"]
    assert skipped[0]["command_returncode"] is None
    assert skipped[0]["requested_device"] == "cuda"
    assert "prediction-producing jobs only" in skipped[0]["reason"]
    assert (run_dir / "protocol_execution_results.json").exists()


def test_protocol_execute_candidate_rejects_artifact_only_skips(tmp_path):
    run_dir = tmp_path / "p2_run"
    manifest = materialize_protocol_run("p2", [ROUTE], run_dir=run_dir)

    with pytest.raises(ValueError, match="cannot be bounded"):
        execute_protocol_jobs(
            manifest,
            protocol_run_manifest_path=run_dir / "protocol_run_manifest.json",
            gate="candidate",
            max_jobs=0,
        )


def test_protocol_execute_candidate_runs_artifact_adapter_before_dependent_eval(tmp_path, monkeypatch):
    import hust_bci_er.evaluation.protocols.runner as runner

    run_dir = tmp_path / "p2_run"
    manifest = materialize_protocol_run("p2", [ROUTE], run_dir=run_dir)

    class Completed:
        returncode = 0
        stdout = "{}"
        stderr = ""

    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: Completed())

    results = execute_protocol_jobs(
        manifest,
        protocol_run_manifest_path=run_dir / "protocol_run_manifest.json",
        gate="candidate",
        allow_artifact_only=True,
    )

    assert results[0]["status"] == "EXECUTED_ARTIFACT"
    assert all(item["status"] == "SKIPPED_DEPENDENCY_FAILED" for item in results[1:])


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
    param_grid = {
        "ea_deformer": [
            {"training.optimizer.lr": 0.0001},
            {"training.optimizer.lr": 0.001},
            {"training.optimizer.lr": 0.01},
        ]
    }
    manifest = materialize_protocol_run(
        "p3",
        [ROUTE],
        run_dir=tmp_path / "p3_run",
        outer_folds=2,
        inner_folds=2,
        param_grids=param_grid,
    )
    inner_job = next(job for job in manifest["jobs"] if job["stage"] == "inner_select")

    split_payload = yaml.safe_load((tmp_path / "p3_run" / inner_job["split_manifest_path"]).read_text(encoding="utf-8"))

    assert "param_index" not in split_payload
    assert split_payload["param_indices"] == [0, 1, 2]
    assert "seed" not in split_payload
    assert split_payload["seeds"] == [123, 124, 125]
    assert "checkpoint.pt" in inner_job["expected_artifacts"]
    assert inner_job["param_overrides"] == {"training.optimizer.lr": 0.0001}
    final_job = next(job for job in manifest["jobs"] if job["stage"] == "outer_final_retrain")
    assert len(final_job["selection_artifact_job_ids"]) == 6
    assert all(path.endswith("selection_metrics.json") for path in final_job["selection_artifact_paths"])
    assert "reuse_checkpoint_path" not in final_job or final_job["reuse_checkpoint_path"] is None


def test_p3_materialization_rejects_abstract_multi_param_grid(tmp_path):
    with pytest.raises(ValueError, match="requires concrete parameter overrides"):
        materialize_protocol_run(
            "p3",
            [ROUTE],
            run_dir=tmp_path / "p3_abstract",
            outer_folds=2,
            inner_folds=2,
            grid_sizes={"ea_deformer": 3},
        )


def test_protocol_runner_records_multiple_routes_in_shared_split_contract(tmp_path):
    manifest = materialize_protocol_run("p2", [ROUTE, ROUTE2], run_dir=tmp_path / "p2_multi_route")
    first_job = manifest["jobs"][0]

    split_payload = yaml.safe_load((tmp_path / "p2_multi_route" / first_job["split_manifest_path"]).read_text(encoding="utf-8"))

    assert "route_id" not in split_payload
    assert split_payload["route_ids"] == ["ea_deformer", "sliding_window_eegnet"]
    assert any(job_id.startswith("p2__ea_deformer__") for job_id in split_payload["job_ids"])
    assert any(job_id.startswith("p2__sliding_window_eegnet__") for job_id in split_payload["job_ids"])


def test_protocol_runner_materializes_formal_p1_splits_from_hust_data_root(tmp_path):
    data_root = write_hust_data_root(tmp_path / "data")

    manifest = materialize_protocol_run(
        "p1",
        [ROUTE],
        run_dir=tmp_path / "p1_formal",
        seeds=[42],
        n_folds=3,
        data_root=data_root,
        default_execute_device="cuda",
    )
    job = manifest["jobs"][0]
    split_payload = yaml.safe_load((tmp_path / "p1_formal" / job["split_manifest_path"]).read_text(encoding="utf-8"))

    assert manifest["split_contract_evidence"] == "formal_subject_trial_rows"
    assert manifest["default_execute_device"] == "cuda"
    assert split_payload["status"] == "ready"
    assert split_payload["train_subjects"]
    assert split_payload["val_subjects"]
    assert split_payload["test_subjects"]
    assert set(split_payload["train_subjects"]).isdisjoint(split_payload["val_subjects"])
    assert set(split_payload["train_subjects"]).isdisjoint(split_payload["test_subjects"])
    assert len(split_payload["trial_rows"]) == 12 * 8
    assert {row["split"] for row in split_payload["trial_rows"]} == {"train", "val", "test"}


def test_protocol_runner_materializes_formal_p2_splits_from_hust_data_root(tmp_path):
    data_root = write_hust_data_root(tmp_path / "data", subjects_per_cohort=8)

    manifest = materialize_protocol_run(
        "p2",
        [ROUTE],
        run_dir=tmp_path / "p2_formal",
        data_root=data_root,
        n_holdout_subjects=4,
    )
    split_payload = yaml.safe_load((tmp_path / "p2_formal" / manifest["jobs"][0]["split_manifest_path"]).read_text(encoding="utf-8"))

    assert manifest["split_contract_evidence"] == "formal_subject_trial_rows"
    assert split_payload["status"] == "ready"
    assert len(split_payload["test_subjects"]) == 4
    assert sum(subject.startswith("DEP") for subject in split_payload["test_subjects"]) == 2
    assert sum(subject.startswith("HC") for subject in split_payload["test_subjects"]) == 2
    assert split_payload["train_subjects"]
    assert split_payload["val_subjects"]
    assert any(subject.startswith("DEP") for subject in split_payload["val_subjects"])
    assert any(subject.startswith("HC") for subject in split_payload["val_subjects"])
    assert set(split_payload["train_subjects"]).isdisjoint(split_payload["test_subjects"])
    assert {row["split"] for row in split_payload["trial_rows"]} == {"train", "val", "test"}


def test_protocol_runner_materializes_formal_p3_inner_and_final_splits_from_hust_data_root(tmp_path):
    data_root = write_hust_data_root(tmp_path / "data", subjects_per_cohort=8)

    manifest = materialize_protocol_run(
        "p3",
        [ROUTE],
        run_dir=tmp_path / "p3_formal",
        data_root=data_root,
        outer_folds=2,
        inner_folds=2,
        grid_sizes={"ea_deformer": 1},
    )
    inner_job = next(job for job in manifest["jobs"] if job["stage"] == "inner_select")
    final_job = next(job for job in manifest["jobs"] if job["stage"] == "outer_final_retrain")
    inner_split = yaml.safe_load((tmp_path / "p3_formal" / inner_job["split_manifest_path"]).read_text(encoding="utf-8"))
    final_split = yaml.safe_load((tmp_path / "p3_formal" / final_job["split_manifest_path"]).read_text(encoding="utf-8"))

    assert inner_split["status"] == "ready"
    assert final_split["status"] == "ready"
    assert inner_split["train_subjects"] and inner_split["val_subjects"] and inner_split["test_subjects"]
    assert final_split["train_subjects"] and final_split["val_subjects"] and final_split["test_subjects"]
    assert set(final_split["train_subjects"]).isdisjoint(final_split["test_subjects"])


def test_protocol_execute_runs_artifact_job_first_then_skips_remaining_on_budget(tmp_path, monkeypatch):
    import hust_bci_er.evaluation.protocols.runner as runner

    run_dir = tmp_path / "p2_run"
    manifest = materialize_protocol_run("p2", [ROUTE], run_dir=run_dir)

    class Completed:
        returncode = 0
        stdout = "{}"
        stderr = ""

    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: Completed())

    results = execute_protocol_jobs(
        manifest,
        protocol_run_manifest_path=run_dir / "protocol_run_manifest.json",
        allow_artifact_only=True,
        max_jobs=1,
    )

    assert results[0]["status"] == "EXECUTED_ARTIFACT"
    assert all(item["status"] == "SKIPPED_MAX_JOBS" for item in results[1:])


def test_p2_smoke_executes_holdout_eval_in_test_scope(tmp_path, monkeypatch):
    import hust_bci_er.evaluation.protocols.runner as runner

    run_dir = tmp_path / "p2_run"
    manifest = materialize_protocol_run("p2", [ROUTE], run_dir=run_dir)
    commands: list[list[str]] = []

    class Completed:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run(cmd, *args, **kwargs):
        commands.append([str(item) for item in cmd])
        if "scripts/run_artifact_job.py" in [str(item) for item in cmd]:
            checkpoint = run_dir / "job_runs" / "p2__ea_deformer__train_seed42" / "checkpoint.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")
        return Completed()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    results = execute_protocol_jobs(
        manifest,
        protocol_run_manifest_path=run_dir / "protocol_run_manifest.json",
        gate="smoke",
        allow_artifact_only=True,
        max_jobs=2,
    )

    assert results[0]["status"] == "EXECUTED_ARTIFACT"
    assert results[1]["status"] == "EXECUTED"
    route_command = next(cmd for cmd in commands if "scripts/run_route_job.py" in cmd)
    assert route_command[route_command.index("--mode") + 1] == "candidate"


def test_p3_smoke_executes_outer_final_retrain_in_test_scope(tmp_path, monkeypatch):
    import hust_bci_er.evaluation.protocols.runner as runner

    run_dir = tmp_path / "p3_run"
    manifest = materialize_protocol_run(
        "p3",
        [ROUTE],
        run_dir=run_dir,
        outer_folds=2,
        inner_folds=2,
        grid_sizes={"ea_deformer": 1},
    )
    for path in next(job for job in manifest["jobs"] if job["stage"] == "outer_final_retrain")["selection_artifact_paths"]:
        selection = run_dir / path
        checkpoint = selection.parent / "checkpoint.pt"
        selection.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        selection.write_text(
            json.dumps(
                {
                    "stage": "inner_select",
                    "primary_metric": "exact_single_crop_expected_BA",
                    "metric_value": 0.5,
                    "checkpoint_path": "checkpoint.pt",
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            ),
            encoding="utf-8",
        )
    commands: list[list[str]] = []

    class Completed:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run(cmd, *args, **kwargs):
        commands.append([str(item) for item in cmd])
        return Completed()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    results = execute_protocol_jobs(
        manifest,
        protocol_run_manifest_path=run_dir / "protocol_run_manifest.json",
        gate="smoke",
        max_jobs=1,
    )

    assert any(item["status"] == "EXECUTED" and item["job_id"].endswith("__final") for item in results)
    route_command = next(cmd for cmd in commands if "scripts/run_route_job.py" in cmd)
    assert route_command[route_command.index("--mode") + 1] == "candidate"


def test_run_route_job_dispatches_torch_classifier_with_protocol_split(tmp_path, monkeypatch):
    from scripts import run_route_job
    import hust_bci_er.training.real_adapter as real_adapter

    route = tmp_path / "route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: tmp_real",
                "status: IDEA",
                "dataset_version: train_v1",
                "split_id: base_split",
                "seed: 42",
                "input_window_sec: 10",
                "preprocessing: [zscore]",
                "features: []",
                "model: {name: shallow_conv_net}",
                "adaptation: none",
                "training: {trainer: torch_classifier, job_adapter: torch_classifier, epochs: 1, batch_size: 2, optimizer: {name: adamw, lr: 0.001}, loss: cross_entropy}",
                "inference: {top4: true, crop_policy: single}",
                "evaluation: {protocol: p1_repeated_group_kfold, primary_metric: exact_single_crop_expected_BA}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "protocol"
    split = run_dir / "splits" / "job_split.yaml"
    split.parent.mkdir(parents=True)
    split.write_text("split_id: job_split\ntrain_subjects: [s1]\nval_subjects: [s2]\ntest_subjects: [s3]\ntrial_rows: []\n", encoding="utf-8")
    checkpoint = run_dir / "job_runs" / "train_job" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    protocol = {
        "jobs": [
            {
                "job_id": "job1",
                "route_config": route.as_posix(),
                "split_id": "job_split",
                "split_manifest_path": "splits/job_split.yaml",
                "seed": 123,
                "expected_artifacts": ["predictions.csv"],
                "crop_policy": {"name": "crop3", "selection": "fixed_index", "crop_index": 2},
                "reuse_checkpoint_path": "job_runs/train_job/checkpoint.pt",
            }
        ]
    }
    protocol_path = run_dir / "protocol_run_manifest.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    captured = {}

    class Artifacts:
        run_dir = tmp_path / "out" / "job1"
        manifest_json = run_dir / "manifest.json"

    def fake_run_real_classifier_route(**kwargs):
        captured.update(kwargs)
        Artifacts.run_dir.mkdir(parents=True)
        Artifacts.manifest_json.write_text("{}", encoding="utf-8")
        return Artifacts

    monkeypatch.setattr(real_adapter, "run_real_classifier_route", fake_run_real_classifier_route)

    assert run_route_job.main(
        [
            "--protocol-run",
            protocol_path.as_posix(),
            "--job-id",
            "job1",
            "--output-root",
            (tmp_path / "out").as_posix(),
            "--data-root",
            (tmp_path / "data").as_posix(),
            "--device",
            "cpu",
            "--epochs-override",
            "1",
        ]
    ) == 0
    assert captured["run_mode"] == "candidate"
    assert captured["split_id"] == "job_split"
    assert captured["split_manifest_path"] == split
    assert captured["seed"] == 123
    assert captured["device"] == "cpu"
    assert captured["smoke_epochs"] == 1
    assert captured["crop_policy"]["name"] == "crop3"
    assert captured["reuse_checkpoint_path"] == checkpoint


def test_run_route_job_rejects_missing_declared_reuse_checkpoint(tmp_path, monkeypatch):
    from scripts import run_route_job
    import hust_bci_er.training.real_adapter as real_adapter

    route = tmp_path / "route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: tmp_real",
                "status: IDEA",
                "dataset_version: train_v1",
                "split_id: base_split",
                "seed: 42",
                "input_window_sec: 10",
                "preprocessing: [zscore]",
                "features: []",
                "model: {name: shallow_conv_net}",
                "adaptation: none",
                "training: {trainer: torch_classifier, job_adapter: torch_classifier, epochs: 1, batch_size: 2, optimizer: {name: adamw, lr: 0.001}, loss: cross_entropy}",
                "inference: {top4: true, crop_policy: single}",
                "evaluation: {protocol: p1_repeated_group_kfold, primary_metric: exact_single_crop_expected_BA}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "protocol"
    split = run_dir / "splits" / "job_split.yaml"
    split.parent.mkdir(parents=True)
    split.write_text("split_id: job_split\ntrain_subjects: [s1]\nval_subjects: [s2]\ntest_subjects: [s3]\ntrial_rows: []\n", encoding="utf-8")
    protocol = {
        "jobs": [
            {
                "job_id": "job1",
                "route_config": route.as_posix(),
                "split_id": "job_split",
                "split_manifest_path": "splits/job_split.yaml",
                "seed": 123,
                "expected_artifacts": ["predictions.csv"],
                "reuse_checkpoint_path": "job_runs/train_job/missing_checkpoint.pt",
            }
        ]
    }
    protocol_path = run_dir / "protocol_run_manifest.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    def fail_if_called(**kwargs):
        raise AssertionError("missing reuse checkpoint must fail before adapter dispatch")

    monkeypatch.setattr(real_adapter, "run_real_classifier_route", fail_if_called)

    with pytest.raises(ValueError, match="reuse_checkpoint_path"):
        run_route_job.main(
            [
                "--protocol-run",
                protocol_path.as_posix(),
                "--job-id",
                "job1",
                "--output-root",
                (tmp_path / "out").as_posix(),
            ]
        )


def test_run_route_job_selects_best_p3_params_retrains_and_records_manifest(tmp_path, monkeypatch):
    from scripts import run_route_job
    import hust_bci_er.training.real_adapter as real_adapter

    route = tmp_path / "route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: tmp_real",
                "status: IDEA",
                "dataset_version: train_v1",
                "split_id: base_split",
                "seed: 42",
                "input_window_sec: 10",
                "preprocessing: [zscore]",
                "features: []",
                "model: {name: shallow_conv_net}",
                "adaptation: none",
                "training: {trainer: torch_classifier, job_adapter: torch_classifier, epochs: 1, batch_size: 2, optimizer: {name: adamw, lr: 0.001}, loss: cross_entropy}",
                "inference: {top4: true, crop_policy: single}",
                "evaluation: {protocol: p3_nested_selection, primary_metric: exact_single_crop_expected_BA}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "protocol"
    split = run_dir / "splits" / "job_split.yaml"
    split.parent.mkdir(parents=True)
    split.write_text("split_id: job_split\ntrain_subjects: [s1]\nval_subjects: [s2]\ntest_subjects: [s3]\ntrial_rows: []\n", encoding="utf-8")
    selection_paths = []
    for job_id, param_index, inner_fold, metric in [
        ("inner_p0_f0", 0, 0, 0.5),
        ("inner_p0_f1", 0, 1, 0.7),
        ("inner_p1_f0", 1, 0, 0.8),
        ("inner_p1_f1", 1, 1, 0.6),
    ]:
        job_dir = run_dir / "job_runs" / job_id
        job_dir.mkdir(parents=True)
        checkpoint = job_dir / "checkpoint.pt"
        checkpoint.write_bytes(f"checkpoint-{job_id}".encode("utf-8"))
        selection = job_dir / "selection_metrics.json"
        selection.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "stage": "inner_select",
                    "protocol": "p3_nested_selection",
                    "param_index": param_index,
                    "outer_fold": 0,
                    "inner_fold": inner_fold,
                    "param_overrides": {"training.optimizer.lr": 0.001 * (param_index + 1)},
                    "primary_metric": "exact_single_crop_expected_BA",
                    "metric_value": metric,
                    "checkpoint_path": "checkpoint.pt",
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            ),
            encoding="utf-8",
        )
        selection_paths.append(f"job_runs/{job_id}/selection_metrics.json")
    protocol = {
        "jobs": [
            {
                "job_id": "final",
                "route_config": route.as_posix(),
                "split_id": "job_split",
                "split_manifest_path": "splits/job_split.yaml",
                "seed": 42,
                "stage": "outer_final_retrain",
                "expected_artifacts": ["predictions.csv"],
                "selection_artifact_paths": selection_paths,
            }
        ]
    }
    protocol_path = run_dir / "protocol_run_manifest.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    captured = {}

    class Artifacts:
        run_dir = tmp_path / "out" / "final"
        manifest_json = run_dir / "manifest.json"

    def fake_run_real_classifier_route(**kwargs):
        captured.update(kwargs)
        Artifacts.run_dir.mkdir(parents=True, exist_ok=True)
        Artifacts.manifest_json.write_text("{}", encoding="utf-8")
        return Artifacts

    monkeypatch.setattr(real_adapter, "run_real_classifier_route", fake_run_real_classifier_route)

    assert run_route_job.main(
        [
            "--protocol-run",
            protocol_path.as_posix(),
            "--job-id",
            "final",
            "--output-root",
            (tmp_path / "out").as_posix(),
            "--device",
            "cpu",
        ]
    ) == 0

    assert captured["reuse_checkpoint_path"] is None
    assert captured["reuse_checkpoint_context"] is None
    selected_route = Path(captured["route_config_path"])
    selected_route_payload = yaml.safe_load(selected_route.read_text(encoding="utf-8"))
    assert selected_route_payload["training"]["optimizer"]["lr"] == pytest.approx(0.002)
    patched = json.loads(Artifacts.manifest_json.read_text(encoding="utf-8"))
    assert patched["protocol_selected_artifact"]["selected_param_index"] == 1
    assert patched["protocol_selected_artifact"]["source_job_id"] == "inner_p1_f0"
    assert patched["protocol_selected_artifact"]["selected_param_overrides"] == {"training.optimizer.lr": 0.002}
    assert patched["protocol_selected_artifact"]["inner_checkpoint_reused_for_outer_test"] is False


def test_run_route_job_uses_protocol_default_device_and_data_root(tmp_path, monkeypatch):
    from scripts import run_route_job
    import hust_bci_er.training.real_adapter as real_adapter

    route = tmp_path / "route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: tmp_real",
                "status: IDEA",
                "dataset_version: train_v1",
                "split_id: base_split",
                "seed: 42",
                "input_window_sec: 10",
                "preprocessing: [zscore]",
                "features: []",
                "model: {name: shallow_conv_net}",
                "adaptation: none",
                "training: {trainer: torch_classifier, job_adapter: torch_classifier, epochs: 1, batch_size: 2, optimizer: {name: adamw, lr: 0.001}, loss: cross_entropy}",
                "inference: {top4: true, crop_policy: single}",
                "evaluation: {protocol: p1_repeated_group_kfold, primary_metric: exact_single_crop_expected_BA}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "protocol"
    split = run_dir / "splits" / "job_split.yaml"
    split.parent.mkdir(parents=True)
    split.write_text("split_id: job_split\ntrain_subjects: [s1]\nval_subjects: [s2]\ntest_subjects: [s3]\ntrial_rows: []\n", encoding="utf-8")
    data_root = tmp_path / "data"
    protocol = {
        "default_execute_device": "cuda",
        "data_root": data_root.as_posix(),
        "jobs": [
            {
                "job_id": "job1",
                "route_config": route.as_posix(),
                "split_id": "job_split",
                "split_manifest_path": "splits/job_split.yaml",
                "seed": 123,
                "expected_artifacts": ["predictions.csv"],
            }
        ],
    }
    protocol_path = run_dir / "protocol_run_manifest.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    captured = {}

    class Artifacts:
        run_dir = tmp_path / "out" / "job1"
        manifest_json = run_dir / "manifest.json"

    def fake_run_real_classifier_route(**kwargs):
        captured.update(kwargs)
        Artifacts.run_dir.mkdir(parents=True)
        Artifacts.manifest_json.write_text("{}", encoding="utf-8")
        return Artifacts

    monkeypatch.setattr(real_adapter, "run_real_classifier_route", fake_run_real_classifier_route)

    assert run_route_job.main(
        [
            "--protocol-run",
            protocol_path.as_posix(),
            "--job-id",
            "job1",
            "--output-root",
            (tmp_path / "out").as_posix(),
        ]
    ) == 0
    assert captured["device"] == "cuda"
    assert captured["data_root"] == data_root


def test_protocol_summary_writes_fold_aggregate_outputs(tmp_path):
    run_dir = tmp_path / "p1_done"
    job_run = run_dir / "job_runs" / "job1"
    job_run.mkdir(parents=True)
    (run_dir / "protocol_run_manifest.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id": "job1",
                        "route_id": "route_a",
                        "seed": 42,
                        "fold": 0,
                        "stage": "train_eval",
                        "split_id": "split_a",
                        "expected_artifacts": ["predictions.csv", "manifest.json"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (job_run / "manifest.json").write_text(
        json.dumps({"primary_metric": "exact_single_crop_expected_BA", "metrics": {"exact_single_crop_expected_BA": 0.75}}),
        encoding="utf-8",
    )
    (job_run / "audit_report.json").write_text(json.dumps({"overall": "PASS"}), encoding="utf-8")
    (job_run / "subject_ba.csv").write_text(
        "subject_id,balanced_accuracy,metric_value,n_rows,group_key\ns1,0.75,0.75,8,s1\n",
        encoding="utf-8",
    )

    audit = write_protocol_summary(run_dir)

    assert audit["status"] == "COMPLETE"
    board = (run_dir / "protocol1_board.csv").read_text(encoding="utf-8")
    assert "route_a" in board
    assert "0.75" in board
    assert (run_dir / "protocol1_subject_ba.csv").exists()


def test_protocol_summary_marks_missing_artifact_only_jobs_incomplete(tmp_path):
    run_dir = tmp_path / "p2_incomplete"
    (run_dir / "job_runs" / "p2__route__eval_crop1").mkdir(parents=True)
    (run_dir / "protocol_run_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "p2_pseudo_public_holdout",
                "jobs": [
                    {
                        "job_id": "p2__route__train_seed42",
                        "route_id": "route_a",
                        "seed": 42,
                        "stage": "train_holdout_model",
                        "split_id": "split_a",
                        "expected_artifacts": ["checkpoint.pt", "train_manifest.json"],
                    },
                    {
                        "job_id": "p2__route__eval_crop1",
                        "route_id": "route_a",
                        "seed": 42,
                        "stage": "evaluate_holdout_crop_policy",
                        "split_id": "split_a",
                        "expected_artifacts": ["predictions.csv", "manifest.json"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    eval_run = run_dir / "job_runs" / "p2__route__eval_crop1"
    (eval_run / "manifest.json").write_text(
        json.dumps({"primary_metric": "exact_single_crop_expected_BA", "metrics": {"exact_single_crop_expected_BA": 0.5}}),
        encoding="utf-8",
    )
    (eval_run / "audit_report.json").write_text(json.dumps({"overall": "PASS"}), encoding="utf-8")

    audit = write_protocol_summary(run_dir)

    assert audit["status"] == "INCOMPLETE"
    assert audit["n_artifact_only_jobs"] == 1
    assert "p2__route__train_seed42: checkpoint.pt" in audit["missing_artifacts"]
    assert (run_dir / "protocol2_audit.json").exists()


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


def test_write_run_manifest_records_route_model_kwargs(monkeypatch, tmp_path):
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

    manifest = write_run_manifest(
        route_config_path=Path("configs/routes/models/fixed_crop_ea_fbcnet.yaml"),
        run_dir=run_dir,
        prediction_csv=predictions,
        metrics={"exact_single_crop_expected_BA": 1.0},
        command="test",
    )

    assert manifest["model_name"] == "fbcnet"
    assert manifest["model_kwargs"]["n_bands"] == 9
    assert "name" not in manifest["model_kwargs"]


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
    (root / "pyproject.toml").write_text("[project]\nname = 'tmp'\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# tmp\n", encoding="utf-8")
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


def test_launch_reproducible_sets_hash_seed_for_child():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/launch_reproducible.py",
            "--seed",
            "77",
            "--",
            sys.executable,
            "-c",
            "import os; print(os.environ['PYTHONHASHSEED'])",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "77"


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
