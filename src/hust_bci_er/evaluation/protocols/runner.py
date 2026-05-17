"""Protocol runner contracts for P1/P2/P3 experiments."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from hust_bci_er.audit.environment import capture_environment
from hust_bci_er.audit.manifest import sha256_file
from hust_bci_er.audit.run_manifest import default_checkpoint_selection, environment_lock_payload, git_commit, repo_root_from_route
from hust_bci_er.data.hust_mat_index import hust_mat_trial_index
from hust_bci_er.evaluation.crop_policy import crop_policy_manifest, route_crop_policy_manifest
from hust_bci_er.evaluation.protocols.plans import (
    DEFAULT_CROP_POLICIES,
    DEFAULT_SEEDS,
    build_protocol1_plan,
    build_protocol2_plan,
    build_protocol3_plan,
)
from hust_bci_er.evaluation.protocols.subject_splits import (
    assign_trial_rows_to_split,
    p1_subject_split,
    p2_subject_split,
    p3_outer_inner_split,
    p3_outer_subject_split,
)
from hust_bci_er.training.reproducibility import ReproducibilityConfig, reproducibility_manifest


PROTOCOL_ALIASES = {
    "p1": "p1_repeated_group_kfold",
    "p2": "p2_pseudo_public_holdout",
    "p3": "p3_nested_selection",
}


@dataclass(frozen=True)
class ProtocolJob:
    job_id: str
    protocol: str
    route_id: str
    route_config: str
    stage: str
    seed: int
    dataset_version: str
    split_id: str
    source_split_id: str
    expected_artifacts: tuple[str, ...]
    split_manifest_path: str | None = None
    split_sha256: str | None = None
    fold: int | None = None
    outer_fold: int | None = None
    inner_fold: int | None = None
    param_index: int | None = None
    holdout_seed: int | None = None
    n_holdout_subjects: int | None = None
    outer_seed: int | None = None
    inner_seed: int | None = None
    outer_folds: int | None = None
    inner_folds: int | None = None
    crop_policy: Mapping[str, Any] | None = None
    determinism: Mapping[str, Any] | None = None
    checkpoint_selection: Mapping[str, Any] | None = None
    n_folds: int | None = None
    val_fraction: float | None = None
    reuse_checkpoint_job_id: str | None = None
    reuse_checkpoint_path: str | None = None
    selection_artifact_job_ids: tuple[str, ...] = ()
    selection_artifact_paths: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["expected_artifacts"] = list(self.expected_artifacts)
        data["selection_artifact_job_ids"] = list(self.selection_artifact_job_ids)
        data["selection_artifact_paths"] = list(self.selection_artifact_paths)
        return data


def load_route(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"route config must be a mapping: {path}")
    return data


def route_ids(route_data: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(data["route_id"]) for data in route_data]


def route_job_base(route_path: Path, route_data: Mapping[str, Any], *, seed: int, split_id: str) -> dict[str, Any]:
    return {
        "route_id": str(route_data["route_id"]),
        "route_config": route_path.as_posix(),
        "seed": int(seed),
        "dataset_version": str(route_data["dataset_version"]),
        "split_id": split_id,
        "source_split_id": str(route_data["split_id"]),
        "determinism": reproducibility_manifest(ReproducibilityConfig(seed=int(seed))),
        "checkpoint_selection": default_checkpoint_selection(route_data),
    }


def protocol_split_id(source_split_id: str, suffix: str) -> str:
    return f"{source_split_id}__{suffix}"


def build_protocol_jobs(
    protocol: str,
    route_config_paths: Sequence[Path],
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_folds: int = 5,
    val_fraction: float = 0.2,
    n_holdout_subjects: int = 12,
    holdout_seed: int = 999,
    train_seed: int = 42,
    crop_policies: Sequence[str] = DEFAULT_CROP_POLICIES,
    outer_folds: int = 5,
    inner_folds: int = 3,
    outer_seed: int = 42,
    inner_seed: int = 123,
    grid_sizes: Mapping[str, int] | None = None,
) -> tuple[dict[str, Any], list[ProtocolJob]]:
    protocol_name = PROTOCOL_ALIASES.get(protocol, protocol)
    paths = [Path(path) for path in route_config_paths]
    route_data = [load_route(path) for path in paths]
    routes = route_ids(route_data)
    jobs: list[ProtocolJob] = []

    if protocol_name == "p1_repeated_group_kfold":
        plan = build_protocol1_plan(routes, seeds=seeds, n_folds=n_folds, val_fraction=val_fraction)
        for path, data in zip(paths, route_data):
            for seed in seeds:
                for fold in range(int(n_folds)):
                    route_id = str(data["route_id"])
                    split_id = protocol_split_id(str(data["split_id"]), f"p1_seed{int(seed)}_fold{fold}")
                    jobs.append(
                        ProtocolJob(
                            protocol=protocol_name,
                            job_id=f"p1__{route_id}__seed{int(seed)}__fold{fold}",
                            stage="train_eval",
                            fold=fold,
                            expected_artifacts=("config_snapshot.yaml", "predictions.csv", "score_matrix.csv", "manifest.json", "audit_report.json"),
                            crop_policy=route_crop_policy_manifest(data, seed=int(seed)),
                            n_folds=int(n_folds),
                            val_fraction=float(val_fraction),
                            **route_job_base(path, data, seed=int(seed), split_id=split_id),
                        )
                    )
        return plan.as_dict(), jobs

    if protocol_name == "p2_pseudo_public_holdout":
        plan = build_protocol2_plan(routes, n_holdout_subjects=n_holdout_subjects, holdout_seed=holdout_seed, train_seed=train_seed, crop_policies=crop_policies)
        for path, data in zip(paths, route_data):
            route_id = str(data["route_id"])
            split_id = protocol_split_id(str(data["split_id"]), f"p2_holdout{int(holdout_seed)}_train{int(train_seed)}")
            train_job_id = f"p2__{route_id}__train_seed{int(train_seed)}"
            jobs.append(
                ProtocolJob(
                    protocol=protocol_name,
                    job_id=train_job_id,
                    stage="train_holdout_model",
                    expected_artifacts=("config_snapshot.yaml", "checkpoint.pt", "train_manifest.json"),
                    holdout_seed=int(holdout_seed),
                    n_holdout_subjects=int(n_holdout_subjects),
                    **route_job_base(path, data, seed=int(train_seed), split_id=split_id),
                )
            )
            for policy in crop_policies:
                jobs.append(
                    ProtocolJob(
                        protocol=protocol_name,
                        job_id=f"p2__{route_id}__eval_{policy}",
                        stage="evaluate_holdout_crop_policy",
                        expected_artifacts=("predictions.csv", "score_matrix.csv", "manifest.json", "audit_report.json"),
                        crop_policy=crop_policy_manifest(str(policy), seed=int(train_seed)),
                        reuse_checkpoint_job_id=train_job_id,
                        reuse_checkpoint_path=f"job_runs/{train_job_id}/checkpoint.pt",
                        holdout_seed=int(holdout_seed),
                        n_holdout_subjects=int(n_holdout_subjects),
                        **route_job_base(path, data, seed=int(train_seed), split_id=split_id),
                    )
                )
        return plan.as_dict(), jobs

    if protocol_name == "p3_nested_selection":
        plan = build_protocol3_plan(routes, outer_folds=outer_folds, inner_folds=inner_folds, outer_seed=outer_seed, inner_seed=inner_seed, grid_sizes=grid_sizes)
        sizes = {route_id: int((grid_sizes or {}).get(route_id, 1)) for route_id in routes}
        for path, data in zip(paths, route_data):
            route_id = str(data["route_id"])
            for outer in range(int(outer_folds)):
                selection_job_ids: list[str] = []
                for inner in range(int(inner_folds)):
                    for param_index in range(sizes[route_id]):
                        seed = int(inner_seed) + outer * 1000 + inner * 100 + param_index
                        split_id = protocol_split_id(str(data["split_id"]), f"p3_outer{outer}_inner{inner}")
                        inner_job_id = f"p3__{route_id}__outer{outer}__inner{inner}__param{param_index}"
                        selection_job_ids.append(inner_job_id)
                        jobs.append(
                            ProtocolJob(
                                protocol=protocol_name,
                                job_id=inner_job_id,
                                stage="inner_select",
                                outer_fold=outer,
                                inner_fold=inner,
                                param_index=param_index,
                                expected_artifacts=("config_snapshot.yaml", "checkpoint.pt", "selection_metrics.json", "manifest.json"),
                                outer_seed=int(outer_seed),
                                inner_seed=int(inner_seed),
                                outer_folds=int(outer_folds),
                                inner_folds=int(inner_folds),
                                **route_job_base(path, data, seed=seed, split_id=split_id),
                            )
                        )
                seed = int(outer_seed) + outer
                split_id = protocol_split_id(str(data["split_id"]), f"p3_outer{outer}_final")
                jobs.append(
                    ProtocolJob(
                        protocol=protocol_name,
                        job_id=f"p3__{route_id}__outer{outer}__final",
                        stage="outer_train_eval",
                        outer_fold=outer,
                        expected_artifacts=("config_snapshot.yaml", "predictions.csv", "score_matrix.csv", "manifest.json", "audit_report.json"),
                        crop_policy=route_crop_policy_manifest(data, seed=seed),
                        selection_artifact_job_ids=tuple(selection_job_ids),
                        selection_artifact_paths=tuple(f"job_runs/{job_id}/selection_metrics.json" for job_id in selection_job_ids),
                        outer_seed=int(outer_seed),
                        inner_seed=int(inner_seed),
                        outer_folds=int(outer_folds),
                        inner_folds=int(inner_folds),
                        **route_job_base(path, data, seed=seed, split_id=split_id),
                    )
                )
        return plan.as_dict(), jobs

    raise ValueError(f"unknown protocol: {protocol}")


def route_locks(route_config_paths: Sequence[Path], *, run_dir: Path, root: Path) -> list[dict[str, Any]]:
    snapshot_dir = run_dir / "route_configs"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    locks: list[dict[str, Any]] = []
    for path in route_config_paths:
        path = Path(path).resolve()
        data = load_route(path)
        snapshot = snapshot_dir / path.name
        shutil.copyfile(path, snapshot)
        dataset = root / "configs" / "datasets" / f"{data['dataset_version']}.yaml"
        split = root / "configs" / "splits" / f"{data['split_id']}.yaml"
        locks.append(
            {
                "route_id": str(data["route_id"]),
                "route_config_path": path.relative_to(root).as_posix(),
                "route_config_snapshot_path": snapshot.relative_to(run_dir).as_posix(),
                "route_config_snapshot_sha256": sha256_file(snapshot),
                "dataset_manifest_path": dataset.relative_to(root).as_posix(),
                "dataset_manifest_sha256": sha256_file(dataset),
                "source_split_manifest_path": split.relative_to(root).as_posix(),
                "source_split_manifest_sha256": sha256_file(split),
            }
        )
    return locks


def split_contract_payload(
    jobs: Sequence[ProtocolJob],
    *,
    trial_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not jobs:
        raise ValueError("at least one job is required for a split contract")
    job = jobs[0]
    payload: dict[str, Any] = {
        "split_id": job.split_id,
        "status": "materialized_without_subject_index",
        "description": "Job-specific split contract generated by protocol runner. Fill subject lists and trial_rows before candidate review.",
        "source_split_id": job.source_split_id,
        "protocol": job.protocol,
        "job_ids": [item.job_id for item in jobs],
        "stages": sorted({item.stage for item in jobs}),
        "subject_group_split": True,
        "train_subjects": [],
        "val_subjects": [],
        "test_subjects": [],
        "trial_rows": [],
    }
    route_values = sorted({item.route_id for item in jobs})
    if len(route_values) == 1:
        payload["route_id"] = route_values[0]
    else:
        payload["route_ids"] = route_values
    scalar_or_list_fields = {
        "seed": "seeds",
        "fold": "folds",
        "outer_fold": "outer_folds",
        "inner_fold": "inner_folds",
        "param_index": "param_indices",
    }
    for key, plural_key in scalar_or_list_fields.items():
        values = sorted({getattr(item, key) for item in jobs if getattr(item, key) is not None})
        if len(values) == 1:
            payload[key] = values[0]
        elif len(values) > 1:
            payload[plural_key] = values
    crop_policies = [dict(item.crop_policy) for item in jobs if isinstance(item.crop_policy, Mapping)]
    if crop_policies:
        payload["crop_policies"] = crop_policies
    if trial_rows is not None:
        protocol = str(job.protocol)
        if protocol == "p1_repeated_group_kfold":
            if job.fold is None or job.n_folds is None or job.seed is None:
                raise ValueError("P1 formal split requires fold, n_folds, and seed")
            train_subjects, val_subjects, test_subjects = p1_subject_split(
                trial_rows,
                seed=int(job.seed),
                fold=int(job.fold),
                n_folds=int(job.n_folds),
                val_fraction=float(job.val_fraction if job.val_fraction is not None else 0.2),
            )
            payload.update(
                {
                    "status": "ready",
                    "description": "Formal P1 subject-group split contract generated from the indexed HUST EEG data root.",
                    "train_subjects": train_subjects,
                    "val_subjects": val_subjects,
                    "test_subjects": test_subjects,
                    "trial_rows": assign_trial_rows_to_split(
                        trial_rows,
                        train_subjects=train_subjects,
                        val_subjects=val_subjects,
                        test_subjects=test_subjects,
                    ),
                }
            )
        elif protocol == "p2_pseudo_public_holdout":
            train_subjects, val_subjects, test_subjects = p2_subject_split(
                trial_rows,
                holdout_seed=int(job.holdout_seed) if job.holdout_seed is not None else 999,
                train_seed=int(job.seed),
                n_holdout_subjects=int(job.n_holdout_subjects) if job.n_holdout_subjects is not None else 12,
                val_fraction=float(job.val_fraction if job.val_fraction is not None else 0.2),
            )
            payload.update(
                {
                    "status": "ready",
                    "description": "Formal P2 holdout split contract from the indexed HUST EEG data root.",
                    "train_subjects": train_subjects,
                    "val_subjects": val_subjects,
                    "test_subjects": test_subjects,
                    "trial_rows": assign_trial_rows_to_split(
                        trial_rows,
                        train_subjects=train_subjects,
                        val_subjects=val_subjects,
                        test_subjects=test_subjects,
                    ),
                }
            )
        elif protocol == "p3_nested_selection":
            if job.outer_fold is None:
                raise ValueError("P3 formal split requires outer_fold")
            if job.inner_fold is None:
                train_subjects, val_subjects, test_subjects = p3_outer_subject_split(
                    trial_rows,
                    outer_seed=int(job.outer_seed) if job.outer_seed is not None else 42,
                    outer_fold=int(job.outer_fold),
                    outer_folds=int(job.outer_folds) if job.outer_folds is not None else 5,
                    val_fraction=float(job.val_fraction if job.val_fraction is not None else 0.2),
                )
            else:
                train_subjects, val_subjects, test_subjects = p3_outer_inner_split(
                    trial_rows,
                    outer_seed=int(job.outer_seed) if job.outer_seed is not None else 42,
                    inner_seed=int(job.inner_seed) if job.inner_seed is not None else 123,
                    outer_fold=int(job.outer_fold),
                    inner_fold=int(job.inner_fold),
                    outer_folds=int(job.outer_folds) if job.outer_folds is not None else 5,
                    inner_folds=int(job.inner_folds) if job.inner_folds is not None else 3,
                )
            payload.update(
                {
                    "status": "ready",
                    "description": "Formal P3 nested selection split contract from the indexed HUST EEG data root.",
                    "train_subjects": train_subjects,
                    "val_subjects": val_subjects,
                    "test_subjects": test_subjects,
                    "trial_rows": assign_trial_rows_to_split(
                        trial_rows,
                        train_subjects=train_subjects,
                        val_subjects=val_subjects,
                        test_subjects=test_subjects,
                    ),
                }
            )
        else:
            raise ValueError(f"formal data-root split materialization is not supported for protocol: {protocol}")
    return payload


def materialize_job_splits(
    jobs: Sequence[ProtocolJob],
    *,
    run_dir: Path,
    trial_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[ProtocolJob]:
    split_dir = run_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    jobs_by_split: dict[str, list[ProtocolJob]] = {}
    for job in jobs:
        jobs_by_split.setdefault(job.split_id, []).append(job)

    locks: dict[str, tuple[str, str]] = {}
    for split_id, split_jobs in jobs_by_split.items():
        path = split_dir / f"{split_id}.yaml"
        path.write_text(
            yaml.safe_dump(
                split_contract_payload(split_jobs, trial_rows=trial_rows),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        locks[split_id] = (path.relative_to(run_dir).as_posix(), sha256_file(path))

    return [
        replace(job, split_manifest_path=locks[job.split_id][0], split_sha256=locks[job.split_id][1])
        for job in jobs
    ]


def materialize_protocol_run(
    protocol: str,
    route_config_paths: Sequence[Path],
    *,
    run_dir: Path,
    data_root: Path | None = None,
    default_execute_device: str = "auto",
    **kwargs: Any,
) -> dict[str, Any]:
    if not route_config_paths:
        raise ValueError("at least one route config is required")
    root = repo_root_from_route(Path(route_config_paths[0]).resolve())
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    plan, jobs = build_protocol_jobs(protocol, route_config_paths, **kwargs)
    trial_rows = hust_mat_trial_index(data_root) if data_root is not None else None
    jobs = materialize_job_splits(jobs, run_dir=run_dir, trial_rows=trial_rows)
    experiment_gate_job_ids = [job.job_id for job in jobs if "predictions.csv" in job.expected_artifacts]
    artifact_only_job_ids = [job.job_id for job in jobs if job.job_id not in set(experiment_gate_job_ids)]
    manifest = {
        "runner_schema_version": 1,
        "status": "MATERIALIZED",
        "git_commit": git_commit(root),
        "protocol": plan["protocol"],
        "plan": plan,
        "environment": capture_environment(),
        "environment_lock": environment_lock_payload(root),
        "data_root": str(Path(data_root).resolve()) if data_root is not None else None,
        "default_execute_device": str(default_execute_device),
        "split_contract_evidence": "formal_subject_trial_rows" if trial_rows is not None else "placeholder_contracts",
        "route_locks": route_locks(route_config_paths, run_dir=run_dir, root=root),
        "jobs": [job.as_dict() for job in jobs],
        "experiment_gate_job_ids": experiment_gate_job_ids,
        "artifact_only_job_ids": artifact_only_job_ids,
        "completion_rule": "jobs with predictions.csv must write manifest.json and pass the requested experiment gate; P2 artifact jobs must write reusable checkpoints; P3 inner_select jobs must write checkpoint plus selection_metrics.json before dependent final jobs select and reuse the best checkpoint",
    }
    (run_dir / "protocol_run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _mode_for_job(gate: str, job: Mapping[str, Any]) -> str:
    """Derive the route-job run_mode from the execution gate and stage.

    Smoke gate uses ``full_subjects`` mode to avoid candidate-only
    constraints (test-only prediction scope, epoch override rejection).
    P2 holdout and P3 final evaluation are the exceptions: even smoke
    diagnostics must evaluate the materialized holdout/test split rather
    than validation subjects, while still auditing with the smoke gate.
    Candidate gate always uses ``candidate`` mode.
    """
    if gate == "smoke":
        if str(job.get("stage", "")) in {"evaluate_holdout_crop_policy", "outer_train_eval"}:
            return "candidate"
        return "full_subjects"
    return "candidate"


def _execute_artifact_job(
    job: Mapping[str, Any],
    *,
    root: Path,
    run_manifest: Path,
    effective_device: str,
    data_root: Path | None,
    epochs_override: int | None,
) -> dict[str, Any]:
    """Execute an artifact-only protocol job through the artifact-job adapter."""
    job_id = str(job.get("job_id", ""))
    run_dir = run_manifest.parent / "job_runs" / job_id

    command = [
        sys.executable,
        "scripts/launch_reproducible.py",
        "--seed",
        str(job["seed"]),
        "--",
        sys.executable,
        "scripts/run_artifact_job.py",
        "--protocol-run",
        str(run_manifest),
        "--job-id",
        job_id,
        "--device",
        effective_device,
    ]
    if data_root:
        command.extend(["--data-root", str(data_root)])
    if epochs_override is not None:
        command.extend(["--epochs-override", str(int(epochs_override))])

    proc = subprocess.run(command, cwd=root, text=True, capture_output=True)
    return {
        "job_id": job_id,
        "route_config": str(job.get("route_config", "")),
        "run_dir": str(run_dir),
        "status": "EXECUTED_ARTIFACT" if proc.returncode == 0 else "FAILED_ARTIFACT",
        "command_returncode": proc.returncode,
        "audit_gate": None,
        "audit_returncode": None,
        "requested_device": effective_device,
        "stdout_tail": proc.stdout[-4000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-4000:] if proc.stderr else "",
    }


def _missing_dependency_artifacts(job: Mapping[str, Any], *, run_manifest: Path) -> list[str]:
    protocol_root = run_manifest.parent
    missing: list[str] = []
    reuse_path = job.get("reuse_checkpoint_path")
    if isinstance(reuse_path, str) and reuse_path and not (protocol_root / reuse_path).exists():
        missing.append(reuse_path)
    for path in job.get("selection_artifact_paths") or []:
        if not isinstance(path, str) or not path:
            continue
        artifact_path = protocol_root / path
        if not artifact_path.exists():
            missing.append(path)
            continue
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            missing.append(f"{path}: invalid JSON")
            continue
        checkpoint_rel = payload.get("checkpoint_path") if isinstance(payload, Mapping) else None
        if not isinstance(checkpoint_rel, str) or not checkpoint_rel:
            missing.append(f"{path}: checkpoint_path")
            continue
        if not (artifact_path.parent / checkpoint_rel).exists():
            missing.append(f"{path}: {checkpoint_rel}")
    return missing


def execute_protocol_jobs(
    manifest: Mapping[str, Any],
    *,
    protocol_run_manifest_path: Path,
    gate: str = "smoke",
    max_jobs: int | None = None,
    data_root: Path | None = None,
    device: str = "auto",
    epochs_override: int | None = None,
    allow_artifact_only: bool = False,
) -> list[dict[str, Any]]:
    """Execute prediction-producing jobs through the stable route-job adapter.

    P2/P3 artifact-only jobs (train_holdout_model, inner_select) are
    executed through the artifact-job adapter for candidate gates, because
    downstream evaluation jobs depend on their reusable checkpoint/selection
    artifacts.  For smoke gates, pass *allow_artifact_only* to exercise those
    artifact stages instead of listing them as skipped diagnostics.
    """
    if gate == "candidate" and epochs_override is not None:
        raise ValueError("epochs_override cannot be used with candidate gate")
    effective_device = str(device if str(device) != "auto" else manifest.get("default_execute_device") or "auto")
    run_manifest = protocol_run_manifest_path.resolve()
    root = next(
        (
            candidate
            for candidate in (run_manifest.parent, *run_manifest.parent.parents)
            if (candidate / "pyproject.toml").exists() and (candidate / "AGENTS.md").exists()
        ),
        None,
    )
    cwd = Path.cwd()
    if root is None and (cwd / "pyproject.toml").exists() and (cwd / "AGENTS.md").exists():
        root = cwd
    if root is None:
        raise ValueError(f"could not find repository root for protocol run: {protocol_run_manifest_path}")
    jobs = [job for job in manifest.get("jobs", []) if isinstance(job, Mapping)]
    runnable = [job for job in jobs if "predictions.csv" in job.get("expected_artifacts", [])]
    artifact_only = [job for job in jobs if "predictions.csv" not in job.get("expected_artifacts", [])]
    if gate == "candidate" and max_jobs is not None and int(max_jobs) < len(jobs):
        raise ValueError("candidate protocol execution cannot be bounded by --max-execute-jobs")
    execute_artifact_jobs = gate == "candidate" or allow_artifact_only
    results: list[dict[str, Any]] = []
    execution_budget = None if max_jobs is None else max(0, int(max_jobs))
    executed_count = 0

    def budget_exhausted() -> bool:
        return execution_budget is not None and executed_count >= execution_budget

    for job in artifact_only:
        if execute_artifact_jobs:
            if budget_exhausted():
                result = {
                    "job_id": str(job.get("job_id", "")),
                    "route_config": str(job.get("route_config", "")),
                    "run_dir": None,
                    "status": "SKIPPED_MAX_JOBS",
                    "reason": "--max-execute-jobs limit reached before this artifact-only job.",
                    "command_returncode": None,
                    "audit_gate": gate,
                    "audit_returncode": None,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "requested_device": effective_device,
                }
            else:
                result = _execute_artifact_job(job, root=root, run_manifest=run_manifest, effective_device=effective_device, data_root=data_root, epochs_override=epochs_override)
                executed_count += 1
        else:
            result = {
                "job_id": str(job.get("job_id", "")),
                "route_config": str(job.get("route_config", "")),
                "run_dir": None,
                "status": "SKIPPED_ARTIFACT_ONLY",
                "reason": "--execute currently runs prediction-producing jobs only; this artifact-only job must be produced by the formal training/selection adapter.",
                "command_returncode": None,
                "audit_gate": gate,
                "audit_returncode": None,
                "stdout_tail": "",
                "stderr_tail": "",
                "requested_device": effective_device,
            }
        results.append(result)
    dependency_failed = any(item.get("status") in {"FAILED_ARTIFACT", "SKIPPED_MAX_JOBS"} for item in results)
    for job in runnable:
        missing_dependencies = _missing_dependency_artifacts(job, run_manifest=run_manifest)
        if budget_exhausted():
            results.append(
                {
                    "job_id": str(job.get("job_id", "")),
                    "route_config": str(job.get("route_config", "")),
                    "run_dir": None,
                    "status": "SKIPPED_MAX_JOBS",
                    "reason": "--max-execute-jobs limit reached before this prediction-producing job.",
                    "command_returncode": None,
                    "audit_gate": gate,
                    "audit_returncode": None,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "requested_device": effective_device,
                }
            )
            continue
        if dependency_failed or missing_dependencies:
            results.append(
                {
                    "job_id": str(job.get("job_id", "")),
                    "route_config": str(job.get("route_config", "")),
                    "run_dir": None,
                    "status": "SKIPPED_DEPENDENCY_FAILED",
                    "reason": "protocol job dependencies are missing or failed: " + ", ".join(missing_dependencies),
                    "command_returncode": None,
                    "audit_gate": gate,
                    "audit_returncode": None,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "requested_device": effective_device,
                }
            )
            continue
        seed = int(job["seed"])
        command = [
            sys.executable,
            "scripts/launch_reproducible.py",
            "--seed",
            str(seed),
            "--",
            sys.executable,
            "scripts/run_route_job.py",
            "--protocol-run",
            str(protocol_run_manifest_path),
            "--job-id",
            str(job["job_id"]),
            "--mode",
            _mode_for_job(gate, job),
            "--device",
            effective_device,
        ]
        resolved_data_root = data_root or manifest.get("data_root")
        if resolved_data_root:
            command.extend(["--data-root", str(resolved_data_root)])
        if epochs_override is not None:
            command.extend(["--epochs-override", str(int(epochs_override))])
        route_config = str(job["route_config"])
        run_dir = protocol_run_manifest_path.resolve().parent / "job_runs" / str(job["job_id"])
        proc = subprocess.run(command, cwd=root, text=True, capture_output=True)
        audit_code: int | None = None
        if proc.returncode == 0:
            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/repo_doctor.py",
                    "experiment",
                    "--route",
                    route_config,
                    "--run",
                    str(run_dir),
                    "--gate",
                    gate,
                ],
                cwd=root,
                text=True,
                capture_output=True,
            )
            audit_code = audit.returncode
            proc_stdout = proc.stdout + audit.stdout
            proc_stderr = proc.stderr + audit.stderr
        else:
            proc_stdout = proc.stdout
            proc_stderr = proc.stderr
        results.append(
            {
                "job_id": str(job["job_id"]),
                "route_config": route_config,
                "run_dir": str(run_dir),
                "status": "EXECUTED",
                "command_returncode": proc.returncode,
                "audit_gate": gate,
                "audit_returncode": audit_code,
                "requested_device": effective_device,
                "stdout_tail": proc_stdout[-4000:],
                "stderr_tail": proc_stderr[-4000:],
            }
        )
        executed_count += 1
    results_path = protocol_run_manifest_path.resolve().parent / "protocol_execution_results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return results
