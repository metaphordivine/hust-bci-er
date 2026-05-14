"""Writers for run manifests that lock experiment artifacts."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from hust_bci_er.audit.environment import capture_environment
from hust_bci_er.audit.manifest import sha256_file
from hust_bci_er.evaluation.crop_policy import route_crop_policy_manifest
from hust_bci_er.training.reproducibility import ReproducibilityConfig, apply_reproducibility


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML document must be a mapping: {path}")
    return data


def repo_root_from_route(route_config_path: Path) -> Path:
    return route_config_path.resolve().parents[3]


def git_commit(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True, timeout=5)
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "UNKNOWN"


def command_string(command: str | Sequence[str]) -> str:
    if isinstance(command, str):
        return command
    return " ".join(str(item) for item in command)


def relative_to_run(path: Path, run_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(run_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"run artifact must stay under run_dir: {path}") from exc


def evidence_manifest_path(path: Path, *, run_dir: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        pass
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"evidence path must stay under run_dir or repository root: {path}") from exc


def default_checkpoint_selection(route_data: Mapping[str, Any]) -> dict[str, Any]:
    training = route_data.get("training") if isinstance(route_data, Mapping) else None
    early = training.get("early_stopping") if isinstance(training, Mapping) else None
    monitor = "val_loss"
    mode = "min"
    if isinstance(early, Mapping):
        monitor = str(early.get("monitor", monitor))
        mode = str(early.get("mode", mode))
    return {
        "rule": "best_monitored_epoch",
        "monitor": monitor,
        "mode": mode,
        "tie_break": "earliest_epoch",
        "restore_best": True,
    }


def environment_lock_payload(root: Path) -> dict[str, Any] | None:
    files = []
    for name in ["environment.lock", "requirements.lock"]:
        path = root / name
        if path.exists():
            files.append({"path": name, "sha256": sha256_file(path)})
    if not files:
        return None
    return {"files": files}


def group_keys_payload(keys: Sequence[str] | None, *, default: Sequence[str] = ("subject_id",)) -> list[str]:
    values = list(default if keys is None else keys)
    if not values or any(not isinstance(key, str) or not key for key in values):
        raise ValueError("group keys must be a non-empty sequence of strings")
    return values


def write_run_manifest(
    *,
    route_config_path: Path,
    run_dir: Path,
    prediction_csv: Path,
    metrics: Mapping[str, float],
    command: str | Sequence[str],
    dataset_manifest_path: Path | None = None,
    split_manifest_path: Path | None = None,
    split_id: str | None = None,
    source_split_id: str | None = None,
    seed: int | None = None,
    source_seed: int | None = None,
    score_matrix_csv: Path | None = None,
    environment: Mapping[str, Any] | None = None,
    determinism: Mapping[str, Any] | None = None,
    checkpoint_selection: Mapping[str, Any] | None = None,
    crop_policy: Mapping[str, Any] | None = None,
    top4_group_keys: Sequence[str] | None = None,
    metric_group_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write ``manifest.json`` for a finished run and return its payload."""
    route_config_path = route_config_path.resolve()
    root = repo_root_from_route(route_config_path)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    route_data = load_yaml_mapping(route_config_path)
    route_id = str(route_data["route_id"])
    source_seed = int(route_data["seed"] if source_seed is None else source_seed)
    seed = int(source_seed if seed is None else seed)
    source_split_id = source_split_id or str(route_data["split_id"])
    split_id = split_id or source_split_id
    snapshot_path = run_dir / "config_snapshot.yaml"
    shutil.copyfile(route_config_path, snapshot_path)

    dataset_manifest_path = dataset_manifest_path or root / "configs" / "datasets" / f"{route_data['dataset_version']}.yaml"
    if split_manifest_path is None:
        if split_id != source_split_id:
            raise ValueError("split_manifest_path is required when split_id overrides the route split_id")
        split_manifest_path = root / "configs" / "splits" / f"{source_split_id}.yaml"
    prediction_csv = prediction_csv.resolve()
    if not prediction_csv.exists():
        raise FileNotFoundError(f"prediction_csv not found: {prediction_csv}")

    primary_metric = str(route_data["evaluation"]["primary_metric"])
    manifest: dict[str, Any] = {
        "audit_schema_version": 2,
        "route_id": route_id,
        "git_commit": git_commit(root),
        "config_path": route_config_path.relative_to(root).as_posix(),
        "config_snapshot_path": snapshot_path.relative_to(run_dir).as_posix(),
        "config_sha256": sha256_file(snapshot_path),
        "dataset_manifest_path": evidence_manifest_path(dataset_manifest_path, run_dir=run_dir, root=root),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "split_manifest_path": evidence_manifest_path(split_manifest_path, run_dir=run_dir, root=root),
        "split_id": split_id,
        "source_split_id": source_split_id,
        "split_sha256": sha256_file(split_manifest_path),
        "seed": seed,
        "source_seed": source_seed,
        "command": command_string(command),
        "primary_metric": primary_metric,
        "metrics": dict(metrics),
        "prediction_csv": relative_to_run(prediction_csv, run_dir),
        "prediction_sha256": sha256_file(prediction_csv),
        "prediction_record_level": "trial",
        "top4_group_keys": group_keys_payload(top4_group_keys),
        "environment": dict(environment or capture_environment()),
        "determinism": dict(determinism or apply_reproducibility(ReproducibilityConfig(seed=seed))),
        "checkpoint_selection": dict(checkpoint_selection or default_checkpoint_selection(route_data)),
        "crop_policy": dict(crop_policy or route_crop_policy_manifest(route_data, seed=seed)),
    }
    if metric_group_keys is not None:
        manifest["metric_group_keys"] = group_keys_payload(metric_group_keys)
    elif top4_group_keys is not None:
        manifest["metric_group_keys"] = manifest["top4_group_keys"]
    lock = environment_lock_payload(root)
    if lock is not None:
        manifest["environment_lock"] = lock
    if score_matrix_csv is not None:
        score_matrix_csv = score_matrix_csv.resolve()
        if not score_matrix_csv.exists():
            raise FileNotFoundError(f"score_matrix_csv not found: {score_matrix_csv}")
        manifest["metric_inputs"] = {
            "score_matrix_csv": relative_to_run(score_matrix_csv, run_dir),
            "score_matrix_sha256": sha256_file(score_matrix_csv),
        }

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
