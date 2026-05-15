from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.training.toy_adapter import run_toy_route  # noqa: E402


def default_run_dir(route: Path) -> Path:
    return ROOT / "outputs" / route.stem / "latest"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a route through a stable training job adapter.")
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--split-id")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    run_dir = args.run_dir or default_run_dir(args.route)
    artifacts = run_toy_route(
        route_config_path=args.route,
        run_dir=run_dir,
        split_id=args.split_id,
        seed=args.seed,
        command=["python", "scripts/train_route.py", "--route", args.route.as_posix(), "--run-dir", run_dir.as_posix()],
    )
    print(
        json.dumps(
            {
                "route": args.route.as_posix(),
                "run_dir": str(artifacts.run_dir),
                "manifest": str(artifacts.manifest_json),
                "primary_metric": artifacts.metric_report["metrics"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

