"""Artifact manifest checks."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any

import yaml


REQUIRED_MANIFEST_FIELDS = {
    "audit_schema_version",
    "route_id",
    "git_commit",
    "config_path",
    "config_snapshot_path",
    "config_sha256",
    "dataset_manifest_path",
    "dataset_manifest_sha256",
    "split_manifest_path",
    "split_id",
    "split_sha256",
    "seed",
    "command",
    "primary_metric",
    "metrics",
    "prediction_csv",
    "prediction_sha256",
}

REQUIRED_TYPES = {
    "audit_schema_version": int,
    "route_id": str,
    "git_commit": str,
    "config_path": str,
    "config_snapshot_path": str,
    "config_sha256": str,
    "dataset_manifest_path": str,
    "dataset_manifest_sha256": str,
    "split_manifest_path": str,
    "split_id": str,
    "split_sha256": str,
    "seed": int,
    "command": str,
    "primary_metric": str,
    "metrics": dict,
    "prediction_csv": str,
    "prediction_sha256": str,
}

HASH_FIELDS = {
    "config_sha256",
    "dataset_manifest_sha256",
    "split_sha256",
    "prediction_sha256",
}

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml_file(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def resolve_run_path(run_dir: Path, value: str, *, field: str, errors: list[str]) -> Path | None:
    resolved = (run_dir / value).resolve()
    if not is_under(resolved, run_dir):
        errors.append(f"manifest {field} must stay under the run directory")
        return None
    return resolved


def resolve_repo_or_run_path(run_dir: Path, root: Path, value: str, *, field: str, repo_subdir: str, errors: list[str]) -> Path | None:
    run_candidate = (run_dir / value).resolve()
    repo_root = (root / repo_subdir).resolve()
    repo_candidate = (root / value).resolve()
    if run_candidate.exists():
        if is_under(run_candidate, run_dir):
            return run_candidate
        errors.append(f"manifest {field} run path escapes the run directory")
        return None
    if repo_candidate.exists():
        if is_under(repo_candidate, repo_root):
            return repo_candidate
        errors.append(f"manifest {field} repo path must stay under {repo_subdir}")
        return None
    if is_under(run_candidate, run_dir):
        return run_candidate
    if is_under(repo_candidate, repo_root):
        return repo_candidate
    errors.append(f"manifest {field} must stay under the run directory or {repo_subdir}")
    return None


def validate_manifest(path: Path, *, root: Path | None = None, route_data: dict[str, Any] | None = None) -> list[str]:
    if not path.exists():
        return [f"manifest not found: {path}"]
    data = load_manifest(path)
    if not isinstance(data, dict):
        return [f"manifest must be a JSON object: {path}"]

    missing = sorted(REQUIRED_MANIFEST_FIELDS.difference(data))
    errors = [f"manifest missing fields: {', '.join(missing)}"] if missing else []

    for key, expected_type in REQUIRED_TYPES.items():
        if key in data and not isinstance(data[key], expected_type):
            errors.append(f"manifest {key} must be {expected_type.__name__}")

    for key in HASH_FIELDS:
        value = data.get(key)
        if isinstance(value, str) and SHA256_RE.fullmatch(value) is None:
            errors.append(f"manifest {key} must be a lowercase sha256 hex string")

    root = root or path.parent
    run_dir = path.parent
    snapshot_data: dict[str, Any] = {}
    config_path = data.get("config_path")
    config_snapshot_path = data.get("config_snapshot_path")
    expected_hash = data.get("config_sha256")
    if isinstance(config_snapshot_path, str) and isinstance(expected_hash, str):
        resolved = resolve_run_path(run_dir, config_snapshot_path, field="config_snapshot_path", errors=errors)
        if resolved is None:
            pass
        elif not resolved.exists():
            errors.append(f"manifest config_snapshot_path not found: {config_snapshot_path}")
        else:
            actual_hash = sha256_file(resolved)
            if actual_hash != expected_hash:
                errors.append("manifest config_sha256 does not match config snapshot")
            else:
                snapshot_data = load_yaml_file(resolved)
    elif isinstance(config_path, str) and isinstance(expected_hash, str):
        resolved = (root / config_path).resolve()
        if not resolved.exists():
            errors.append(f"manifest config_path not found: {config_path}")
        elif sha256_file(resolved) != expected_hash:
            errors.append("manifest config_sha256 does not match current config file")
        else:
            snapshot_data = load_yaml_file(resolved)

    route_basis = snapshot_data or route_data or {}
    if route_basis:
        for key in ["route_id", "split_id", "seed"]:
            if key in data and key in route_basis and data[key] != route_basis[key]:
                errors.append(f"manifest {key} does not match config snapshot")
        route_metric = None
        evaluation = route_basis.get("evaluation")
        if isinstance(evaluation, dict):
            route_metric = evaluation.get("primary_metric")
        if data.get("primary_metric") and route_metric and data["primary_metric"] != route_metric:
            errors.append("manifest primary_metric does not match config snapshot")

    prediction_csv = data.get("prediction_csv")
    prediction_hash = data.get("prediction_sha256")
    if isinstance(prediction_csv, str) and isinstance(prediction_hash, str):
        resolved = resolve_run_path(run_dir, prediction_csv, field="prediction_csv", errors=errors)
        if resolved is None:
            pass
        elif not resolved.exists():
            errors.append(f"manifest prediction_csv not found: {prediction_csv}")
        elif sha256_file(resolved) != prediction_hash:
            errors.append("manifest prediction_sha256 does not match prediction CSV")

    dataset_path = data.get("dataset_manifest_path")
    dataset_hash = data.get("dataset_manifest_sha256")
    if isinstance(dataset_path, str) and isinstance(dataset_hash, str):
        resolved = resolve_repo_or_run_path(run_dir, root, dataset_path, field="dataset_manifest_path", repo_subdir="configs/datasets", errors=errors)
        if resolved is None:
            pass
        elif not resolved.exists():
            errors.append(f"manifest dataset_manifest_path not found: {dataset_path}")
        else:
            if sha256_file(resolved) != dataset_hash:
                errors.append("manifest dataset_manifest_sha256 does not match dataset manifest")
            dataset_data = load_yaml_file(resolved)
            expected_dataset = route_basis.get("dataset_version") if route_basis else None
            if expected_dataset and dataset_data.get("dataset_version") != expected_dataset:
                errors.append("manifest dataset_manifest_path does not match route dataset_version")

    split_hash = data.get("split_sha256")
    split_manifest_path = data.get("split_manifest_path")
    if isinstance(split_manifest_path, str) and isinstance(split_hash, str):
        resolved = resolve_repo_or_run_path(run_dir, root, split_manifest_path, field="split_manifest_path", repo_subdir="configs/splits", errors=errors)
        if resolved is None:
            pass
        elif not resolved.exists():
            errors.append(f"manifest split_manifest_path not found: {split_manifest_path}")
        else:
            if sha256_file(resolved) != split_hash:
                errors.append("manifest split_sha256 does not match split manifest")
            split_data = load_yaml_file(resolved)
            expected_split = data.get("split_id")
            if expected_split and split_data.get("split_id") != expected_split:
                errors.append("manifest split_manifest_path does not match manifest split_id")
    elif route_data is not None and isinstance(split_hash, str):
        split_id = route_data.get("split_id")
        if isinstance(split_id, str) and split_id:
            resolved = root / "configs" / "splits" / f"{split_id}.yaml"
            if resolved.exists() and sha256_file(resolved) != split_hash:
                errors.append("manifest split_sha256 does not match route split manifest")

    return errors
