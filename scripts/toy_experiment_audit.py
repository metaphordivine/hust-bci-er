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


def write_summary(route: Path, run_dir: Path) -> Path:
    audit_report_path = run_dir / "audit_report.json"
    manifest_path = run_dir / "manifest.json"
    route_data = load_mapping(route)
    audit_report = load_mapping(audit_report_path)
    summary_path = ROOT / "reports" / "route_summaries" / f"{route_data['route_id']}_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        render_route_summary(route_data=route_data, audit_report=audit_report, manifest_path=manifest_path, root=ROOT),
        encoding="utf-8",
    )
    return summary_path


def audit_once(route: Path, run_dir: Path, *, gate: str) -> dict:
    report = run_audit(route, run_dir, gate=gate)
    write_reports(run_dir, report)
    return report


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
    summary_path: Path | None = None
    if report["overall"] != "PASS" and args.gate == "candidate":
        summary_path = write_summary(args.route, args.run_dir)
        try:
            report = audit_once(args.route, args.run_dir, gate=args.gate)
        finally:
            if summary_path.exists():
                summary_path.unlink()
            if summary_path.parent.exists() and not any(summary_path.parent.iterdir()):
                summary_path.parent.rmdir()

    print(json.dumps({"route_id": report["route_id"], "gate": report["gate"], "overall": report["overall"], "run_dir": str(args.run_dir.resolve())}, ensure_ascii=False))
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
