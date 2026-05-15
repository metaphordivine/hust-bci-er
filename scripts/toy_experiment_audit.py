from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.audit.summary import load_mapping, render_route_summary  # noqa: E402
from hust_bci_er.training.toy_adapter import run_toy_route  # noqa: E402
from scripts.audit_experiment import run_audit, write_reports  # noqa: E402


def write_summary(route: Path, run_dir: Path, *, summary_dir: Path) -> Path:
    audit_report_path = run_dir / "audit_report.json"
    manifest_path = run_dir / "manifest.json"
    route_data = load_mapping(route)
    audit_report = load_mapping(audit_report_path)
    summary_path = summary_dir / f"{route_data['route_id']}_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        render_route_summary(route_data=route_data, audit_report=audit_report, manifest_path=manifest_path, root=ROOT),
        encoding="utf-8",
    )
    return summary_path


def audit_once(route: Path, run_dir: Path, *, gate: str, summary_dir: Path | None = None) -> dict:
    report = run_audit(route, run_dir, gate=gate, summary_dir=summary_dir)
    write_reports(run_dir, report)
    return report


def failed_only_missing_summary(report: dict) -> bool:
    failing = [
        check
        for check in report.get("checks", [])
        if isinstance(check, dict) and check.get("status") == "FAIL"
    ]
    return bool(failing) and all(check.get("rule_id") == "SUMMARY_EXISTS" for check in failing)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the toy route end-to-end and prove candidate audit can pass.")
    parser.add_argument("--route", type=Path, default=ROOT / "configs" / "routes" / "models" / "toy_eegnet.yaml")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / "toy_eegnet" / "ci")
    parser.add_argument("--gate", choices=["candidate", "smoke"], default="candidate")
    args = parser.parse_args(argv)

    run_toy_route(
        route_config_path=args.route,
        run_dir=args.run_dir,
        command=["python", "scripts/toy_experiment_audit.py", "--route", args.route.as_posix(), "--run-dir", args.run_dir.as_posix()],
    )
    report = audit_once(args.route, args.run_dir, gate=args.gate)
    if args.gate == "candidate" and report["overall"] != "PASS" and failed_only_missing_summary(report):
        # Candidate audit intentionally requires a route summary, but the toy
        # CI smoke must not create run-specific files under tracked reports/.
        # Keep the proof summary inside this run directory so interruptions
        # cannot leave accidental changes in the repository tree.
        summary_dir = args.run_dir / "route_summaries"
        write_summary(args.route, args.run_dir, summary_dir=summary_dir)
        report = audit_once(args.route, args.run_dir, gate=args.gate, summary_dir=summary_dir)
        if report["overall"] == "PASS":
            write_summary(args.route, args.run_dir, summary_dir=summary_dir)

    print(json.dumps({"route_id": report["route_id"], "gate": report["gate"], "overall": report["overall"], "run_dir": str(args.run_dir.resolve())}, ensure_ascii=False))
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
