"""Artifact manifest checks."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_MANIFEST_FIELDS = {
    "route_id",
    "git_commit",
    "config_path",
    "config_sha256",
    "dataset_manifest_sha256",
    "split_id",
    "split_sha256",
    "seed",
    "command",
    "primary_metric",
}


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(path: Path) -> list[str]:
    if not path.exists():
        return [f"manifest not found: {path}"]
    data = load_manifest(path)
    missing = sorted(REQUIRED_MANIFEST_FIELDS.difference(data))
    return [f"manifest missing fields: {', '.join(missing)}"] if missing else []

