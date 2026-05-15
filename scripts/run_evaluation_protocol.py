from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.evaluation.protocols.runner import execute_protocol_jobs, materialize_protocol_run  # noqa: E402
from scripts.plan_evaluation_protocol import parse_grid_size  # noqa: E402


def default_run_dir(protocol: str) -> Path:
    return ROOT / "outputs" / "protocol_runs" / protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize an auditable P1/P2/P3 protocol runner manifest.")
    parser.add_argument("--protocol", choices=["p1", "p2", "p3"], required=True)
    parser.add_argument("--route-config", action="append", type=Path, required=True, help="Route config path. Can be repeated.")
    parser.add_argument("--run-dir", type=Path, help="Output directory for protocol_run_manifest.json.")
    parser.add_argument("--seed", action="append", type=int, default=[], help="P1 seed. Can be repeated.")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-holdout-subjects", type=int, default=12)
    parser.add_argument("--holdout-seed", type=int, default=999)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--crop-policy", action="append", default=[], help="P2 crop policy. Can be repeated.")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--outer-seed", type=int, default=42)
    parser.add_argument("--inner-seed", type=int, default=123)
    parser.add_argument("--grid-size", action="append", type=parse_grid_size, default=[], help="P3 route grid size, route_id=size.")
    parser.add_argument("--execute", action="store_true", help="Execute prediction-producing jobs with the supported route adapter.")
    parser.add_argument("--execute-gate", choices=["smoke", "candidate"], default="smoke")
    parser.add_argument("--max-execute-jobs", type=int)
    args = parser.parse_args(argv)

    kwargs = {}
    if args.protocol == "p1":
        kwargs = {"seeds": args.seed or (42, 123, 456, 789, 1024), "n_folds": args.n_folds}
    elif args.protocol == "p2":
        kwargs = {
            "n_holdout_subjects": args.n_holdout_subjects,
            "holdout_seed": args.holdout_seed,
            "train_seed": args.train_seed,
            "crop_policies": args.crop_policy or ("crop1", "crop2", "crop3", "crop4", "crop5", "random", "worst"),
        }
    else:
        kwargs = {
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "outer_seed": args.outer_seed,
            "inner_seed": args.inner_seed,
            "grid_sizes": dict(args.grid_size),
        }

    run_dir = args.run_dir or default_run_dir(args.protocol)
    manifest = materialize_protocol_run(
        args.protocol,
        args.route_config,
        run_dir=run_dir,
        **kwargs,
    )
    payload = {"protocol": manifest["protocol"], "jobs": len(manifest["jobs"]), "run_dir": str(run_dir.resolve())}
    if args.execute:
        results = execute_protocol_jobs(
            manifest,
            protocol_run_manifest_path=run_dir.resolve() / "protocol_run_manifest.json",
            gate=args.execute_gate,
            max_jobs=args.max_execute_jobs,
        )
        payload["executed_jobs"] = len(results)
        payload["failed_jobs"] = sum(1 for item in results if item["command_returncode"] != 0 or item["audit_returncode"] not in {0, None})
        if payload["failed_jobs"]:
            print(json.dumps(payload, ensure_ascii=False))
            return 1
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
