from __future__ import annotations

import sys
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_REQUIRED_STATUSES = {"CANDIDATE", "PROMOTED", "REJECTED", "ARCHIVED"}
REQUIRED_SUMMARY_TOKENS = {
    "route_id:",
    "route_status:",
    "audit_decision:",
    "gate:",
    "primary_metric:",
    "decision:",
    "reproduce:",
}
REQUIRED_EVIDENCE_FIELDS = {
    "audit_report_path",
    "manifest_path",
    "manifest_sha256",
    "primary_metric_value",
}


def route_files() -> list[Path]:
    return sorted((ROOT / "configs" / "routes" / "models").glob("*.yaml"))


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def summary_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-").strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip().strip("`")
    return fields


def main() -> int:
    errors: list[str] = []
    summary_root = ROOT / "reports" / "route_summaries"
    routes = {load_yaml(path).get("route_id"): load_yaml(path) for path in route_files()}

    for route_id, data in routes.items():
        if not isinstance(route_id, str):
            continue
        summary = summary_root / f"{route_id}_summary.md"
        if data.get("status") in SUMMARY_REQUIRED_STATUSES and not summary.exists():
            errors.append(f"summary required for {route_id}: {summary.relative_to(ROOT)}")

    for summary in summary_root.glob("*_summary.md"):
        route_id = summary.name.removesuffix("_summary.md")
        text = summary.read_text(encoding="utf-8", errors="ignore")
        if route_id not in routes:
            errors.append(f"summary has no matching route config: {summary.relative_to(ROOT)}")
            continue
        if f"route_id: {route_id}" not in text:
            errors.append(f"summary route_id does not match file name: {summary.relative_to(ROOT)}")
        missing = sorted(token for token in REQUIRED_SUMMARY_TOKENS if token not in text)
        if missing:
            errors.append(f"summary missing tokens {missing}: {summary.relative_to(ROOT)}")
        fields = summary_fields(text)
        if routes[route_id].get("status") in SUMMARY_REQUIRED_STATUSES:
            missing_fields = sorted(field for field in REQUIRED_EVIDENCE_FIELDS if not fields.get(field))
            if missing_fields:
                errors.append(f"summary missing evidence fields {missing_fields}: {summary.relative_to(ROOT)}")
        audit_report_path = fields.get("audit_report_path")
        if audit_report_path:
            report_path = (ROOT / audit_report_path).resolve()
            try:
                report_path.relative_to(ROOT.resolve())
            except ValueError:
                errors.append(f"summary audit_report_path escapes repository: {summary.relative_to(ROOT)}")
                continue
            if not report_path.exists():
                errors.append(f"summary audit_report_path does not exist: {summary.relative_to(ROOT)}")
                continue
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"summary audit_report_path is not valid JSON: {summary.relative_to(ROOT)} ({exc})")
                continue
            if report.get("route_id") != route_id:
                errors.append(f"summary audit_report route_id mismatch: {summary.relative_to(ROOT)}")
            if fields.get("gate") and report.get("gate") != fields["gate"]:
                errors.append(f"summary audit_report gate mismatch: {summary.relative_to(ROOT)}")
            if fields.get("audit_decision") and report.get("overall") != fields["audit_decision"]:
                errors.append(f"summary audit_report decision mismatch: {summary.relative_to(ROOT)}")

    if errors:
        print("Summary consistency check failed:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("Summary consistency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
