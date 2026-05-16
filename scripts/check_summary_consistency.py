from __future__ import annotations

import hashlib
import json
import math
import shlex
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

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
REPRODUCE_REPO_RELATIVE_FLAGS = {"--route", "--run", "--config", "--summary-dir"}
REPRODUCE_PATH_FLAGS = REPRODUCE_REPO_RELATIVE_FLAGS | {"--data-root"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def resolve_repo_path(value: str) -> Path | None:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        return None
    return path


def clean_reproduce_token(value: str) -> str:
    return value.strip().strip("\"'")


def is_env_reference(value: str) -> bool:
    cleaned = clean_reproduce_token(value)
    return cleaned.startswith("$") or (cleaned.startswith("%") and cleaned.endswith("%"))


def is_absolute_path_token(value: str) -> bool:
    cleaned = clean_reproduce_token(value)
    if not cleaned or is_env_reference(cleaned):
        return False
    return PureWindowsPath(cleaned).is_absolute() or PurePosixPath(cleaned).is_absolute()


def reproduce_path_values(reproduce: str) -> list[tuple[str, str]]:
    try:
        tokens = shlex.split(reproduce, posix=False)
    except ValueError:
        return [("__parse_error__", reproduce)]

    values: list[tuple[str, str]] = []
    idx = 0
    while idx < len(tokens):
        token = clean_reproduce_token(tokens[idx])
        if token in REPRODUCE_PATH_FLAGS:
            value = clean_reproduce_token(tokens[idx + 1]) if idx + 1 < len(tokens) else ""
            values.append((token, value))
            idx += 2
            continue
        for flag in REPRODUCE_PATH_FLAGS:
            prefix = f"{flag}="
            if token.startswith(prefix):
                values.append((flag, clean_reproduce_token(token[len(prefix):])))
                break
        idx += 1
    return values


def reproduce_path_errors(reproduce: str) -> list[str]:
    errors: list[str] = []
    for flag, value in reproduce_path_values(reproduce):
        if flag == "__parse_error__":
            errors.append("summary reproduce command could not be parsed")
            continue
        if not value:
            errors.append(f"summary reproduce command missing value for {flag}")
            continue
        if is_absolute_path_token(value):
            errors.append("summary reproduce command contains local absolute path")
            continue
        if flag in REPRODUCE_REPO_RELATIVE_FLAGS and is_env_reference(value):
            errors.append(f"summary reproduce {flag} must be repo-relative, not an environment reference")
            continue
        if flag in REPRODUCE_REPO_RELATIVE_FLAGS and resolve_repo_path(value) is None:
            errors.append(f"summary reproduce {flag} escapes repository")
    return errors


def parse_numeric(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def validate_summary_manifest_binding(summary: Path, route_id: str, fields: dict[str, str], errors: list[str]) -> None:
    manifest_path_value = fields.get("manifest_path")
    if not manifest_path_value:
        return
    manifest_path = resolve_repo_path(manifest_path_value)
    if manifest_path is None:
        errors.append(f"summary manifest_path escapes repository: {summary.relative_to(ROOT)}")
        return
    if not manifest_path.exists():
        errors.append(f"summary manifest_path does not exist: {summary.relative_to(ROOT)}")
        return
    if not manifest_path.is_file():
        errors.append(f"summary manifest_path is not a file: {summary.relative_to(ROOT)}")
        return

    manifest_sha256 = fields.get("manifest_sha256")
    if manifest_sha256:
        if len(manifest_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in manifest_sha256):
            errors.append(f"summary manifest_sha256 is not lowercase sha256: {summary.relative_to(ROOT)}")
        elif sha256_file(manifest_path) != manifest_sha256:
            errors.append(f"summary manifest_sha256 mismatch: {summary.relative_to(ROOT)}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"summary manifest_path is not valid JSON: {summary.relative_to(ROOT)} ({exc})")
        return
    if not isinstance(manifest, dict):
        errors.append(f"summary manifest_path is not a JSON object: {summary.relative_to(ROOT)}")
        return
    if manifest.get("route_id") != route_id:
        errors.append(f"summary manifest route_id mismatch: {summary.relative_to(ROOT)}")

    manifest_primary_metric = manifest.get("primary_metric")
    if not isinstance(manifest_primary_metric, str) or not manifest_primary_metric:
        errors.append(f"summary manifest primary_metric missing or invalid: {summary.relative_to(ROOT)}")
        return
    primary_metric = manifest_primary_metric
    if fields.get("primary_metric") and fields["primary_metric"] != primary_metric:
        errors.append(f"summary manifest primary_metric mismatch: {summary.relative_to(ROOT)}")
    metrics = manifest.get("metrics")
    primary_metric_value = fields.get("primary_metric_value")
    if primary_metric_value:
        parsed = parse_numeric(primary_metric_value)
        if parsed is None:
            errors.append(f"summary primary_metric_value is not numeric: {summary.relative_to(ROOT)}")
        elif not isinstance(metrics, dict):
            errors.append(f"summary manifest metrics missing or invalid: {summary.relative_to(ROOT)}")
        elif not isinstance(primary_metric, str) or primary_metric not in metrics:
            errors.append(f"summary manifest primary metric value missing: {summary.relative_to(ROOT)}")
        else:
            expected = parse_numeric(str(metrics[primary_metric]))
            if expected is None:
                errors.append(f"summary manifest primary metric is not numeric: {summary.relative_to(ROOT)}")
            elif abs(parsed - expected) > 1e-9:
                errors.append(f"summary primary_metric_value mismatch: {summary.relative_to(ROOT)}")


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
        reproduce = fields.get("reproduce", "")
        for reproduce_error in reproduce_path_errors(reproduce):
            errors.append(f"{reproduce_error}: {summary.relative_to(ROOT)}")
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

        validate_summary_manifest_binding(summary, route_id, fields, errors)

    if errors:
        print("Summary consistency check failed:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("Summary consistency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
