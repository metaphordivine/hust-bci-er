"""Protocol runner contracts for P1/P2/P3 experiments."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from hust_bci_er.audit.environment import capture_environment
from hust_bci_er.audit.manifest import sha256_file
from hust_bci_er.audit.run_manifest import default_checkpoint_selection, environment_lock_payload, git_commit, repo_root_from_route
from hust_bci_er.evaluation.crop_policy import crop_policy_manifest, route_crop_policy_manifest
from hust_bci_er.evaluation.protocols.plans import (
    DEFAULT_CROP_POLICIES,
    DEFAULT_SEEDS,
    build_protocol1_plan,
    build_protocol2_plan,
    build_protocol3_plan,
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
    crop_policy: Mapping[str, Any] | None = None
    determinism: Mapping[str, Any] | None = None
    checkpoint_selection: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["expected_artifacts"] = list(self.expected_artifacts)
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
        plan = build_protocol1_plan(routes, seeds=seeds, n_folds=n_folds)
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
                            **route_job_base(path, data, seed=int(seed), split_id=split_id),
                        )
                    )
        return plan.as_dict(), jobs

    if protocol_name == "p2_pseudo_public_holdout":
        plan = build_protocol2_plan(routes, n_holdout_subjects=n_holdout_subjects, holdout_seed=holdout_seed, train_seed=train_seed, crop_policies=crop_policies)
        for path, data in zip(paths, route_data):
            route_id = str(data["route_id"])
            split_id = protocol_split_id(str(data["split_id"]), f"p2_holdout{int(holdout_seed)}_train{int(train_seed)}")
            jobs.append(
                ProtocolJob(
                    protocol=protocol_name,
                    job_id=f"p2__{route_id}__train_seed{int(train_seed)}",
                    stage="train_holdout_model",
                    expected_artifacts=("config_snapshot.yaml", "checkpoint.local", "train_manifest.json"),
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
                for inner in range(int(inner_folds)):
                    for param_index in range(sizes[route_id]):
                        seed = int(inner_seed) + outer * 1000 + inner * 100 + param_index
                        split_id = protocol_split_id(str(data["split_id"]), f"p3_outer{outer}_inner{inner}")
                        jobs.append(
                            ProtocolJob(
                                protocol=protocol_name,
                                job_id=f"p3__{route_id}__outer{outer}__inner{inner}__param{param_index}",
                                stage="inner_select",
                                outer_fold=outer,
                                inner_fold=inner,
                                param_index=param_index,
                                expected_artifacts=("config_snapshot.yaml", "selection_metrics.json", "manifest.json"),
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


def split_contract_payload(jobs: Sequence[ProtocolJob]) -> dict[str, Any]:
    if not jobs:
        raise ValueError("at least one job is required for a split contract")
    job = jobs[0]
    payload: dict[str, Any] = {
        "split_id": job.split_id,
        "status": "materialized_without_subject_index",
        "description": "Job-specific split contract generated by protocol runner. Fill subject lists and trial_rows before candidate review.",
        "source_split_id": job.source_split_id,
        "protocol": job.protocol,
        "route_id": job.route_id,
        "job_ids": [item.job_id for item in jobs],
        "stages": sorted({item.stage for item in jobs}),
        "subject_group_split": True,
        "train_subjects": [],
        "val_subjects": [],
        "test_subjects": [],
        "trial_rows": [],
    }
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
    return payload


def materialize_job_splits(jobs: Sequence[ProtocolJob], *, run_dir: Path) -> list[ProtocolJob]:
    split_dir = run_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    jobs_by_split: dict[str, list[ProtocolJob]] = {}
    for job in jobs:
        jobs_by_split.setdefault(job.split_id, []).append(job)

    locks: dict[str, tuple[str, str]] = {}
    for split_id, split_jobs in jobs_by_split.items():
        path = split_dir / f"{split_id}.yaml"
        path.write_text(yaml.safe_dump(split_contract_payload(split_jobs), sort_keys=False, allow_unicode=True), encoding="utf-8")
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
    **kwargs: Any,
) -> dict[str, Any]:
    if not route_config_paths:
        raise ValueError("at least one route config is required")
    root = repo_root_from_route(Path(route_config_paths[0]).resolve())
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    plan, jobs = build_protocol_jobs(protocol, route_config_paths, **kwargs)
    jobs = materialize_job_splits(jobs, run_dir=run_dir)
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
        "route_locks": route_locks(route_config_paths, run_dir=run_dir, root=root),
        "jobs": [job.as_dict() for job in jobs],
        "experiment_gate_job_ids": experiment_gate_job_ids,
        "artifact_only_job_ids": artifact_only_job_ids,
        "completion_rule": "jobs with predictions.csv must write manifest.json and pass the requested experiment gate; artifact-only train/selection jobs must write their expected artifacts without requiring prediction CSV",
    }
    (run_dir / "protocol_run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
