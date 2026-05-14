"""Route summary draft generation from audit artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from hust_bci_er.audit.manifest import sha256_file


def load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping: {path}")
    return data


def render_route_summary(*, route_data: Mapping[str, Any], audit_report: Mapping[str, Any], manifest_path: Path | None = None) -> str:
    route_id = str(route_data.get("route_id") or audit_report.get("route_id"))
    metric = route_data.get("evaluation", {}).get("primary_metric") if isinstance(route_data.get("evaluation"), Mapping) else audit_report.get("primary_metric", "")
    manifest_data: dict[str, Any] = {}
    if manifest_path is not None and manifest_path.exists():
        manifest_data = load_mapping(manifest_path)
    metric_value = ""
    for source in (audit_report, manifest_data):
        metrics = source.get("metrics") if isinstance(source, Mapping) else None
        if isinstance(metrics, Mapping) and metric in metrics:
            metric_value = str(metrics[metric])
            break
    lines = [
        f"# {route_id} Summary",
        "",
        f"route_id: {route_id}",
        f"route_status: {route_data.get('status', '')}",
        f"audit_decision: {audit_report.get('overall', '')}",
        f"gate: {audit_report.get('gate', '')}",
        f"primary_metric: {metric}",
        f"primary_metric_value: {metric_value}",
        f"decision: {audit_report.get('overall', '')}",
        f"reproduce: python scripts/repo_doctor.py experiment --route {audit_report.get('route_config', '')} --run {audit_report.get('run_dir', '')} --gate {audit_report.get('gate', '')}",
        f"dataset: {route_data.get('dataset_version', '')}",
        f"split: {route_data.get('split_id', '')}",
        f"seed: {route_data.get('seed', '')}",
        f"protocol: {route_data.get('evaluation', {}).get('protocol', '') if isinstance(route_data.get('evaluation'), Mapping) else ''}",
        "risk notes: generated draft; review before committing.",
    ]
    if manifest_path is not None and manifest_path.exists():
        lines.extend(
            [
                f"audit_report_path: {audit_report.get('run_dir', '')}/audit_report.json",
                f"manifest_path: {manifest_path.as_posix()}",
                f"manifest_sha256: {sha256_file(manifest_path)}",
            ]
        )
    return "\n".join(lines) + "\n"
