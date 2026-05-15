from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def fast() -> int:
    commands = [
        [sys.executable, "scripts/check_repo_conventions.py"],
        [sys.executable, "scripts/scan_no_leakage.py"],
        [sys.executable, "scripts/check_registry_consistency.py"],
        [sys.executable, "scripts/check_route_registry.py"],
        [sys.executable, "scripts/validate_route.py", "--all"],
        [sys.executable, "scripts/update_route_board.py", "--check"],
        [sys.executable, "scripts/check_summary_consistency.py"],
        [sys.executable, "-m", "compileall", "-q", "src", "scripts"],
        [sys.executable, "-m", "pytest", "tests", "-q", "--basetemp=.pytest_tmp"],
    ]
    for cmd in commands:
        code = run_step(cmd)
        if code != 0:
            return code
    return 0


def experiment(route: Path, run_dir: Path | None, gate: str) -> int:
    code = run_step([sys.executable, "scripts/validate_route.py", str(route)])
    if code != 0:
        return code
    cmd = [sys.executable, "scripts/audit_experiment.py", "--route", str(route), "--gate", gate]
    if run_dir is not None:
        cmd.extend(["--run", str(run_dir)])
    return run_step(cmd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fast")

    check_board = sub.add_parser("check-board")
    check_board.add_argument("--write", action="store_true")

    experiment_parser = sub.add_parser("experiment")
    experiment_parser.add_argument("--route", type=Path, required=True)
    experiment_parser.add_argument("--run", type=Path)
    experiment_parser.add_argument("--gate", choices=["smoke", "diagnostic", "candidate", "promoted"], default="candidate")

    args = parser.parse_args(argv)
    if args.command == "fast":
        return fast()
    if args.command == "check-board":
        cmd = [sys.executable, "scripts/update_route_board.py"]
        if not args.write:
            cmd.append("--check")
        return run_step(cmd)
    if args.command == "experiment":
        return experiment(args.route, args.run, args.gate)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
