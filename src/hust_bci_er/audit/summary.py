"""Route summary draft generation from audit artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from hust_bci_er.audit.manifest import sha256_file


ROOT = Path(__file__).resolve().parents[3]


def load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping: {path}")
    return data


def repo_relative_path(path: Path | str, *, root: Path = ROOT, field: str = "path") -> str:
    return resolve_repo_path(path, root=root, field=field).relative_to(root.resolve()).as_posix()


def resolve_repo_path(path: Path | str, *, root: Path = ROOT, field: str = "path") -> Path:
    raw = Path(path)
    resolved = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} must be inside repository: {path}") from exc
    return resolved


def render_route_summary(
    *,
    route_data: Mapping[str, Any],
    audit_report: Mapping[str, Any],
    manifest_path: Path | None = None,
    root: Path = ROOT,
) -> str:
    route_id = str(route_data.get("route_id") or audit_report.get("route_id"))
    metric = route_data.get("evaluation", {}).get("primary_metric") if isinstance(route_data.get("evaluation"), Mapping) else audit_report.get("primary_metric", "")
    manifest_data: dict[str, Any] = {}
    resolved_manifest_path: Path | None = None
    manifest_repo_path = ""
    if manifest_path is not None:
        resolved_manifest_path = resolve_repo_path(manifest_path, root=root, field="manifest_path")
        manifest_repo_path = resolved_manifest_path.relative_to(root.resolve()).as_posix()
        if not resolved_manifest_path.exists():
            raise ValueError(f"manifest_path does not exist: {manifest_repo_path}")
        manifest_data = load_mapping(resolved_manifest_path)
    metric_value = ""
    for source in (audit_report, manifest_data):
        metrics = source.get("metrics") if isinstance(source, Mapping) else None
        if isinstance(metrics, Mapping) and metric in metrics:
            metric_value = str(metrics[metric])
            break
    run_dir = str(audit_report.get("run_dir", ""))
    run_dir_for_reproduce = repo_relative_path(run_dir, root=root, field="run_dir") if run_dir else ""
    route_config = str(audit_report.get("route_config", ""))
    route_config_for_reproduce = repo_relative_path(route_config, root=root, field="route_config") if route_config else ""
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
        f"reproduce: python scripts/repo_doctor.py experiment --route {route_config_for_reproduce} --run {run_dir_for_reproduce} --gate {audit_report.get('gate', '')}",
        f"dataset: {route_data.get('dataset_version', '')}",
        f"split: {route_data.get('split_id', '')}",
        f"seed: {route_data.get('seed', '')}",
        f"protocol: {route_data.get('evaluation', {}).get('protocol', '') if isinstance(route_data.get('evaluation'), Mapping) else ''}",
        "risk notes: generated draft; review before committing.",
    ]
    if resolved_manifest_path is not None:
        audit_report_path = Path(run_dir) / "audit_report.json"
        lines.extend(
            [
                f"audit_report_path: {repo_relative_path(audit_report_path, root=root, field='audit_report_path')}",
                f"manifest_path: {manifest_repo_path}",
                f"manifest_sha256: {sha256_file(resolved_manifest_path)}",
            ]
        )
    return "\n".join(lines) + "\n"
