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
        [sys.executable, "scripts/validate_route.py", "--all"],
        [sys.executable, "scripts/update_route_board.py", "--check"],
        [sys.executable, "-m", "pytest", "tests", "-q"],
    ]
    for cmd in commands:
        code = run_step(cmd)
        if code != 0:
            return code
    return 0


def experiment(route: Path, run_dir: Path, gate: str) -> int:
    code = run_step([sys.executable, "scripts/validate_route.py", str(route)])
    if code != 0:
        return code
    return run_step([sys.executable, "scripts/audit_experiment.py", "--route", str(route), "--run", str(run_dir), "--gate", gate])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fast")

    check_board = sub.add_parser("check-board")
    check_board.add_argument("--write", action="store_true")

    experiment_parser = sub.add_parser("experiment")
    experiment_parser.add_argument("--route", type=Path, required=True)
    experiment_parser.add_argument("--run", type=Path, required=True)
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
