"""Promotion audit helper utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_PROMOTION_FIELDS = {
    "route_id",
    "promoted_from_run",
    "candidate_audit_report",
    "primary_metric",
    "comparison_baseline",
    "risk_review",
    "no_leakage_review",
    "decision",
    "reviewer",
    "date",
}
REQUIRED_CANDIDATE_RULES = {
    "MANIFEST_VALID",
    "PRIMARY_METRIC_RECOMPUTE",
    "PRIMARY_METRIC_REPORTED",
    "RUN_DATASET_EVIDENCE_VALID",
    "RUN_SPLIT_EVIDENCE_VALID",
    "RUN_SPLIT_EVIDENCE_CONSISTENT",
    "RUN_REPRODUCIBILITY_LOCKED",
}
REQUIRED_TOP4_RULES = {
    "PREDICTION_TOP4_RANKING",
    "PREDICTION_TOP4_BINARY",
    "PREDICTION_TRIAL_ID_UNIQUE",
    "PREDICTION_TOP4_TRUTH_BALANCE",
}


def parse_key_value_markdown(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip().lstrip("-").strip()
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip().strip("`")
    return fields


def check_promotion_fields(fields: Mapping[str, str]) -> list[str]:
    return [f"missing promotion field: {key}" for key in sorted(REQUIRED_PROMOTION_FIELDS) if not fields.get(key)]


def check_candidate_audit(path: Path, *, route_id: str) -> list[str]:
    if not path.exists():
        return [f"candidate audit report not found: {path}"]
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if report.get("route_id") != route_id:
        errors.append("candidate audit route_id mismatch")
    if report.get("gate") != "candidate":
        errors.append("candidate audit gate must be candidate")
    if report.get("overall") != "PASS":
        errors.append("candidate audit must be PASS")
    checks = report.get("checks")
    if not isinstance(checks, list):
        errors.append("candidate audit must include a checks list")
        return errors
    statuses = {str(item.get("rule_id")): item.get("status") for item in checks if isinstance(item, Mapping)}
    required = set(REQUIRED_CANDIDATE_RULES)
    if "PREDICTION_TOP4_RANKING" in statuses or "PREDICTION_TOP4_GROUPS" in statuses:
        required.update(REQUIRED_TOP4_RULES)
    missing_or_failed = sorted(rule for rule in required if statuses.get(rule) != "PASS")
    if missing_or_failed:
        errors.append("candidate audit missing passing critical rules: " + ", ".join(missing_or_failed))
    return errors
