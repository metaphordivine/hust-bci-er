from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from hust_bci_er.audit.manifest import load_manifest, validate_manifest  # noqa: E402
from hust_bci_er.config.registry import AUDIT_DECISIONS  # noqa: E402
from hust_bci_er.config.schema import validate_route_config  # noqa: E402
from hust_bci_er.contracts.prediction import prediction_schema  # noqa: E402
from hust_bci_er.evaluation.metrics import balanced_accuracy  # noqa: E402


AuditCheck = dict[str, Any]
GATES = {"smoke", "diagnostic", "candidate", "promoted"}
STRICT_GATES = {"candidate", "promoted"}


def add_check(
    checks: list[AuditCheck],
    *,
    rule_id: str,
    severity: str,
    status: str,
    message: str,
    fix: str = "",
    decision_if_fail: str | None = None,
) -> None:
    checks.append(
        {
            "rule_id": rule_id,
            "severity": severity,
            "status": status,
            "message": message,
            "fix": fix,
            "decision_if_fail": decision_if_fail,
        }
    )


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"route config must be a mapping: {path}")
    return data


def resolve_from_root(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def route_uses_top4(route_data: dict[str, Any]) -> bool:
    inference = route_data.get("inference")
    return isinstance(inference, dict) and inference.get("top4") is True


def add_warn_or_fail(
    checks: list[AuditCheck],
    *,
    gate: str,
    fail_gate: set[str],
    rule_id: str,
    message: str,
    fix: str,
    decision_if_fail: str = "BLOCKED",
) -> None:
    if gate in fail_gate:
        add_check(
            checks,
            rule_id=rule_id,
            severity="ERROR",
            status="FAIL",
            message=message,
            fix=fix,
            decision_if_fail=decision_if_fail,
        )
    else:
        add_check(checks, rule_id=rule_id, severity="WARN", status="WARN", message=message, fix=fix)


def check_prediction_csv(path: Path, checks: list[AuditCheck], *, route_data: dict[str, Any], gate: str) -> None:
    if not path.exists():
        add_check(
            checks,
            rule_id="PREDICTION_FILE_EXISTS",
            severity="ERROR",
            status="FAIL",
            message=f"prediction file is missing: {path}",
            fix="Write the prediction CSV listed in manifest.json or update the manifest.",
            decision_if_fail="DIAGNOSTIC_ONLY",
        )
        return

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = set(reader.fieldnames or [])

    if not rows:
        add_check(
            checks,
            rule_id="PREDICTION_NONEMPTY",
            severity="ERROR",
            status="FAIL",
            message="prediction CSV has no rows",
            fix="Regenerate predictions for the audited run.",
            decision_if_fail="DIAGNOSTIC_ONLY",
        )
        return
    add_check(checks, rule_id="PREDICTION_NONEMPTY", severity="INFO", status="PASS", message=f"prediction rows: {len(rows)}")

    schema = prediction_schema(fields)
    if schema["score"] is None:
        add_check(
            checks,
            rule_id="PREDICTION_SCORE_COLUMN",
            severity="WARN",
            status="WARN",
            message="prediction CSV has no standard score column",
            fix="Use one of: score, y_score, probability, logit.",
        )
    else:
        add_check(checks, rule_id="PREDICTION_SCORE_COLUMN", severity="INFO", status="PASS", message="score column found")

    top4_required = route_uses_top4(route_data)
    if schema["subject_id"] and schema["trial_id"] and schema["pred_top4"]:
        bad_groups: list[str] = []
        groups: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            groups.setdefault(str(row[schema["subject_id"]]), []).append(row)
        for subject_id, group in groups.items():
            n_pred = sum(int(float(row[schema["pred_top4"]])) for row in group)
            if len(group) != 8 or n_pred != 4:
                bad_groups.append(f"{subject_id}: rows={len(group)}, pred_top4={n_pred}")
        if bad_groups:
            add_check(
                checks,
                rule_id="PREDICTION_TOP4_GROUPS",
                severity="ERROR",
                status="FAIL",
                message="invalid Top-4 groups: " + "; ".join(bad_groups[:5]),
                fix="Ensure every subject has exactly 8 trials and exactly 4 positive Top-4 predictions.",
                decision_if_fail="DIAGNOSTIC_ONLY",
            )
        else:
            add_check(checks, rule_id="PREDICTION_TOP4_GROUPS", severity="INFO", status="PASS", message="Top-4 groups are valid")
    else:
        add_warn_or_fail(
            checks,
            gate=gate,
            fail_gate={"diagnostic", "candidate", "promoted"} if top4_required else {"promoted"},
            rule_id="PREDICTION_TOP4_GROUPS",
            message="Top-4 group columns are not present",
            fix="For Top-4 audits, include subject_id or user_id, trial_id, and pred_top4.",
            decision_if_fail="BLOCKED",
        )

    if schema["y_true"] and schema["y_pred"]:
        y_true = []
        y_pred = []
        for row in rows:
            y_true.append(int(float(row[schema["y_true"]])))
            y_pred.append(int(float(row[schema["y_pred"]])))
        score = balanced_accuracy(np.array(y_true), np.array(y_pred))
        add_check(checks, rule_id="METRIC_RECOMPUTE_BA", severity="INFO", status="PASS", message=f"recomputed BA: {score:.6f}")
    else:
        add_warn_or_fail(
            checks,
            gate=gate,
            fail_gate=STRICT_GATES,
            rule_id="METRIC_RECOMPUTE_BA",
            message="y_true/y_pred columns are not present; BA recompute skipped",
            fix="For labeled validation runs, include y_true and y_pred in the prediction CSV.",
            decision_if_fail="DIAGNOSTIC_ONLY",
        )


def enforce_gate_strictness(checks: list[AuditCheck], *, gate: str) -> None:
    if gate in STRICT_GATES and any(check["status"] == "WARN" for check in checks):
        add_check(
            checks,
            rule_id="GATE_WARN_STRICTNESS",
            severity="ERROR",
            status="FAIL",
            message=f"{gate} gate does not allow unresolved WARN checks",
            fix="Resolve WARN checks or rerun under a less strict gate.",
            decision_if_fail="BLOCKED",
        )


def overall_decision(checks: list[AuditCheck], *, gate: str) -> str:
    enforce_gate_strictness(checks, gate=gate)
    for decision in ["REJECT", "BLOCKED", "DIAGNOSTIC_ONLY"]:
        if any(check["status"] == "FAIL" and check.get("decision_if_fail") == decision for check in checks):
            return decision
    if any(check["status"] == "WARN" for check in checks):
        return "WARN"
    return "PASS"


def write_reports(run_dir: Path, report: dict[str, Any]) -> None:
    json_path = run_dir / "audit_report.json"
    md_path = run_dir / "audit_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Experiment Audit Report",
        "",
        f"- route_id: `{report['route_id']}`",
        f"- gate: `{report['gate']}`",
        f"- audit_decision: `{report['overall']}`",
        f"- run_dir: `{report.get('run_dir') or ''}`",
        "",
        "| rule_id | severity | status | message | fix |",
        "|---|---|---|---|---|",
    ]
    for check in report["checks"]:
        message = str(check["message"]).replace("|", "\\|")
        fix = str(check.get("fix") or "").replace("|", "\\|")
        lines.append(f"| `{check['rule_id']}` | {check['severity']} | {check['status']} | {message} | {fix} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(route_path: Path, run_dir: Path | None, *, gate: str) -> dict[str, Any]:
    if gate not in GATES:
        raise ValueError(f"unknown gate: {gate}")
    checks: list[AuditCheck] = []
    route_path = route_path.resolve()

    try:
        route_data = load_yaml(route_path)
        route_errors = validate_route_config(route_data, path=route_path)
    except Exception as exc:
        route_data = {}
        route_errors = [str(exc)]

    route_id = str(route_data.get("route_id") or route_path.stem)
    if route_errors:
        add_check(
            checks,
            rule_id="ROUTE_SCHEMA",
            severity="CRITICAL",
            status="FAIL",
            message="; ".join(route_errors),
            fix="Fix the route config and rerun validate_route.py.",
            decision_if_fail="REJECT",
        )
    else:
        add_check(checks, rule_id="ROUTE_SCHEMA", severity="INFO", status="PASS", message="route config is valid")

    if run_dir is None:
        add_warn_or_fail(
            checks,
            gate=gate,
            fail_gate=STRICT_GATES,
            rule_id="RUN_DIRECTORY",
            message="no run directory provided; experiment artifact checks skipped",
            fix="Pass --run for a finished experiment audit.",
            decision_if_fail="BLOCKED",
        )
    else:
        run_dir = run_dir.resolve()
        if not run_dir.exists():
            add_check(
                checks,
                rule_id="RUN_DIRECTORY",
                severity="ERROR",
                status="FAIL",
                message=f"run directory is missing: {run_dir}",
                fix="Provide an existing run output directory.",
                decision_if_fail="BLOCKED",
            )
        else:
            add_check(checks, rule_id="RUN_DIRECTORY", severity="INFO", status="PASS", message="run directory exists")
            manifest_path = run_dir / "manifest.json"
            manifest_errors = validate_manifest(manifest_path, root=ROOT, route_data=route_data)
            if manifest_errors:
                decision = "DIAGNOSTIC_ONLY" if any("not found" in item for item in manifest_errors) else "BLOCKED"
                add_check(
                    checks,
                    rule_id="MANIFEST_VALID",
                    severity="ERROR",
                    status="FAIL",
                    message="; ".join(manifest_errors),
                    fix="Regenerate the run manifest from the actual route config and run artifacts.",
                    decision_if_fail=decision,
                )
            else:
                add_check(checks, rule_id="MANIFEST_VALID", severity="INFO", status="PASS", message="manifest is valid")
                manifest = load_manifest(manifest_path)
                prediction_csv = manifest.get("prediction_csv")
                if isinstance(prediction_csv, str) and prediction_csv:
                    check_prediction_csv(resolve_from_root(prediction_csv, base=run_dir), checks, route_data=route_data, gate=gate)
                else:
                    add_warn_or_fail(
                        checks,
                        gate=gate,
                        fail_gate=STRICT_GATES,
                        rule_id="PREDICTION_FILE_DECLARED",
                        message="manifest has no prediction_csv field",
                        fix="Add prediction_csv to manifest when predictions are available.",
                        decision_if_fail="BLOCKED",
                    )

            summary_path = ROOT / "reports" / "route_summaries" / f"{route_id}_summary.md"
            if summary_path.exists():
                add_check(checks, rule_id="SUMMARY_EXISTS", severity="INFO", status="PASS", message=f"summary found: {summary_path.relative_to(ROOT)}")
            else:
                add_warn_or_fail(
                    checks,
                    gate=gate,
                    fail_gate={"diagnostic", "candidate", "promoted"},
                    rule_id="SUMMARY_EXISTS",
                    message=f"route summary is missing: {summary_path.relative_to(ROOT)}",
                    fix="Write one concise route summary before treating this as a finished experiment.",
                    decision_if_fail="BLOCKED",
                )

    decision = overall_decision(checks, gate=gate)
    if decision not in AUDIT_DECISIONS:
        raise RuntimeError(f"unknown audit decision: {decision}")
    return {
        "route_id": route_id,
        "route_config": str(route_path),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "gate": gate,
        "overall": decision,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--gate", choices=sorted(GATES), default="candidate")
    args = parser.parse_args(argv)

    report = run_audit(args.route, args.run, gate=args.gate)
    if args.run is not None and args.run.exists():
        write_reports(args.run.resolve(), report)

    print(json.dumps({"route_id": report["route_id"], "gate": report["gate"], "overall": report["overall"]}, ensure_ascii=False))
    if report["overall"] in {"REJECT", "BLOCKED", "DIAGNOSTIC_ONLY"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
