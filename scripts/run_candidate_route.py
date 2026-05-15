from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.audit.summary import load_mapping, render_route_summary  # noqa: E402
from hust_bci_er.training.real_adapter import run_real_classifier_route  # noqa: E402
from scripts.audit_experiment import run_audit, write_reports  # noqa: E402


def _command(args: argparse.Namespace) -> list[str]:
    cmd = [
        "python",
        "scripts/run_candidate_route.py",
        "--route",
        args.route.as_posix(),
        "--run-dir",
        args.run_dir.as_posix(),
    ]
    if args.split_id is not None:
        cmd.extend(["--split-id", args.split_id])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.epochs_override is not None:
        cmd.extend(["--epochs-override", str(args.epochs_override)])
    if args.device != "auto":
        cmd.extend(["--device", args.device])
    if args.data_root is not None:
        cmd.extend(["--data-root", args.data_root.as_posix()])
    return cmd


def _audit_once(route: Path, run_dir: Path, *, gate: str = "candidate") -> dict:
    report = run_audit(route, run_dir, gate=gate)
    write_reports(run_dir, report)
    return report


def _failed_only_missing_summary(report: dict) -> bool:
    failing = [
        check
        for check in report.get("checks", [])
        if isinstance(check, dict) and check.get("status") == "FAIL"
    ]
    return bool(failing) and all(check.get("rule_id") == "SUMMARY_EXISTS" for check in failing)


def _write_summary(route: Path, run_dir: Path, summary_dir: Path) -> Path:
    route_data = load_mapping(route)
    audit_report = load_mapping(run_dir / "audit_report.json")
    summary_path = summary_dir / f"{route_data['route_id']}_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        render_route_summary(
            route_data=route_data,
            audit_report=audit_report,
            manifest_path=run_dir / "manifest.json",
            root=ROOT,
        ),
        encoding="utf-8",
    )
    return summary_path


def _write_summary_placeholder(route: Path, run_dir: Path, summary_dir: Path) -> Path:
    route_data = load_mapping(route)
    route_id = str(route_data["route_id"])
    metric = route_data.get("evaluation", {}).get("primary_metric", "") if isinstance(route_data.get("evaluation"), dict) else ""
    summary_path = summary_dir / f"{route_id}_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(
            [
                f"# {route_id} Summary",
                "",
                f"route_id: {route_id}",
                f"route_status: {route_data.get('status', '')}",
                "audit_decision: PENDING",
                "gate: candidate",
                f"primary_metric: {metric}",
                "primary_metric_value:",
                "decision: PENDING",
                f"reproduce: python scripts/repo_doctor.py experiment --route {route.as_posix()} --run {run_dir.as_posix()} --gate candidate",
                f"dataset: {route_data.get('dataset_version', '')}",
                f"split: {route_data.get('split_id', '')}",
                f"seed: {route_data.get('seed', '')}",
                f"protocol: {route_data.get('evaluation', {}).get('protocol', '') if isinstance(route_data.get('evaluation'), dict) else ''}",
                "risk notes: temporary placeholder for candidate audit existence check; overwritten after PASS.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_path


def _write_run_local_summary_placeholder(route: Path, run_dir: Path) -> Path:
    return _write_summary_placeholder(route, run_dir, run_dir / "route_summaries")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a real HUST EEG route and complete candidate audit handoff.")
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs-override", type=int, help="Override route epochs for bounded validation.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args(argv)
    summary_dir = ROOT / "reports" / "route_summaries"

    command = _command(args)
    artifacts = run_real_classifier_route(
        route_config_path=args.route,
        run_dir=args.run_dir,
        run_mode="candidate",
        split_id=args.split_id,
        seed=args.seed,
        command=command,
        data_root=args.data_root,
        smoke_epochs=args.epochs_override,
        device=args.device,
    )

    report = _audit_once(args.route, args.run_dir)
    summary_path: Path | None = None
    if report["overall"] != "PASS" and _failed_only_missing_summary(report):
        _write_run_local_summary_placeholder(args.route, args.run_dir)
        report = run_audit(
            args.route,
            args.run_dir,
            gate="candidate",
            summary_dir=args.run_dir / "route_summaries",
            allow_run_local_summary=True,
        )
        write_reports(args.run_dir, report)
    if report["overall"] == "PASS":
        summary_path = _write_summary(args.route, args.run_dir, summary_dir)
        report = _audit_once(args.route, args.run_dir)

    print(
        json.dumps(
            {
                "route": args.route.as_posix(),
                "run_dir": str(artifacts.run_dir),
                "manifest": str(artifacts.manifest_json),
                "audit_overall": report["overall"],
                "summary": str(summary_path) if summary_path is not None else None,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
