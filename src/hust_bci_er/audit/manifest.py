"""Artifact manifest checks."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any


REQUIRED_MANIFEST_FIELDS = {
    "audit_schema_version",
    "route_id",
    "git_commit",
    "config_path",
    "config_snapshot_path",
    "config_sha256",
    "dataset_manifest_sha256",
    "split_id",
    "split_sha256",
    "seed",
    "command",
    "primary_metric",
    "prediction_csv",
    "prediction_sha256",
}


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest(path: Path, *, root: Path | None = None, route_data: dict[str, Any] | None = None) -> list[str]:
    if not path.exists():
        return [f"manifest not found: {path}"]
    data = load_manifest(path)
    if not isinstance(data, dict):
        return [f"manifest must be a JSON object: {path}"]

    missing = sorted(REQUIRED_MANIFEST_FIELDS.difference(data))
    errors = [f"manifest missing fields: {', '.join(missing)}"] if missing else []

    if route_data is not None:
        for key in ["route_id", "split_id", "seed"]:
            if key in data and key in route_data and data[key] != route_data[key]:
                errors.append(f"manifest {key} does not match route config")
        route_metric = None
        evaluation = route_data.get("evaluation")
        if isinstance(evaluation, dict):
            route_metric = evaluation.get("primary_metric")
        if data.get("primary_metric") and route_metric and data["primary_metric"] != route_metric:
            errors.append("manifest primary_metric does not match route config")

    root = root or path.parent
    run_dir = path.parent
    config_path = data.get("config_path")
    config_snapshot_path = data.get("config_snapshot_path")
    expected_hash = data.get("config_sha256")
    if isinstance(config_snapshot_path, str) and isinstance(expected_hash, str):
        resolved = (run_dir / config_snapshot_path).resolve()
        if not resolved.exists():
            errors.append(f"manifest config_snapshot_path not found: {config_snapshot_path}")
        else:
            actual_hash = sha256_file(resolved)
            if actual_hash != expected_hash:
                errors.append("manifest config_sha256 does not match config snapshot")
    elif isinstance(config_path, str) and isinstance(expected_hash, str):
        resolved = (root / config_path).resolve()
        if not resolved.exists():
            errors.append(f"manifest config_path not found: {config_path}")
        elif sha256_file(resolved) != expected_hash:
            errors.append("manifest config_sha256 does not match current config file")

    prediction_csv = data.get("prediction_csv")
    prediction_hash = data.get("prediction_sha256")
    if isinstance(prediction_csv, str) and isinstance(prediction_hash, str):
        resolved = (run_dir / prediction_csv).resolve()
        if not resolved.exists():
            errors.append(f"manifest prediction_csv not found: {prediction_csv}")
        elif sha256_file(resolved) != prediction_hash:
            errors.append("manifest prediction_sha256 does not match prediction CSV")

    split_hash = data.get("split_sha256")
    split_manifest_path = data.get("split_manifest_path")
    if isinstance(split_manifest_path, str) and isinstance(split_hash, str):
        resolved = (run_dir / split_manifest_path).resolve()
        if not resolved.exists():
            resolved = (root / split_manifest_path).resolve()
        if not resolved.exists():
            errors.append(f"manifest split_manifest_path not found: {split_manifest_path}")
        elif sha256_file(resolved) != split_hash:
            errors.append("manifest split_sha256 does not match split manifest")
    elif route_data is not None and isinstance(split_hash, str):
        split_id = route_data.get("split_id")
        if isinstance(split_id, str) and split_id:
            resolved = root / "configs" / "splits" / f"{split_id}.yaml"
            if resolved.exists() and sha256_file(resolved) != split_hash:
                errors.append("manifest split_sha256 does not match route split manifest")

    return errors
