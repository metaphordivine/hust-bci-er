from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.evaluation.protocols.runner import execute_protocol_jobs, materialize_protocol_run  # noqa: E402
from hust_bci_er.evaluation.protocols.params import load_param_grid  # noqa: E402
from scripts.plan_evaluation_protocol import parse_grid_size, parse_route_path  # noqa: E402


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
    parser.add_argument("--param-grid", action="append", type=parse_route_path, default=[], help="P3 route parameter grid YAML, route_id=path. Overrides --grid-size count and materializes param_overrides.")
    parser.add_argument("--data-root", type=Path, help="HUST EEG .mat data root. When provided for P1, job split contracts include formal subject/trial rows.")
    parser.add_argument("--device", default="auto", help="Torch device passed to executed route jobs.")
    parser.add_argument("--execute", action="store_true", help="Execute prediction-producing jobs with the supported route adapter.")
    parser.add_argument("--execute-gate", choices=["smoke", "candidate"], default="smoke")
    parser.add_argument("--execute-epochs-override", type=int, help="Bound executed jobs for diagnostic runs. Do not use with candidate evidence.")
    parser.add_argument("--max-execute-jobs", type=int)
    parser.add_argument("--allow-artifact-only", action="store_true", help="For smoke execution, also run P2/P3 artifact-only jobs (train_holdout, inner_select). Candidate execution always runs required artifact jobs.")
    args = parser.parse_args(argv)
    if args.execute_gate == "candidate" and args.execute_epochs_override is not None:
        parser.error("--execute-epochs-override cannot be used with --execute-gate candidate")

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
        param_grids = {route_id: load_param_grid(path) for route_id, path in args.param_grid}
        grid_sizes = dict(args.grid_size)
        abstract_multi_grid_routes = [
            route_id
            for route_id, size in grid_sizes.items()
            if int(size) > 1 and route_id not in param_grids
        ]
        if abstract_multi_grid_routes:
            parser.error(
                "--grid-size > 1 is dry-run planning only for P3 materialization; "
                "use --param-grid route_id=search_space.yaml for: "
                + ", ".join(sorted(abstract_multi_grid_routes))
            )
        grid_sizes.update({route_id: len(candidates) for route_id, candidates in param_grids.items()})
        kwargs = {
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "outer_seed": args.outer_seed,
            "inner_seed": args.inner_seed,
            "grid_sizes": grid_sizes,
            "param_grids": param_grids,
        }

    run_dir = args.run_dir or default_run_dir(args.protocol)
    manifest = materialize_protocol_run(
        args.protocol,
        args.route_config,
        run_dir=run_dir,
        data_root=args.data_root,
        default_execute_device=args.device,
        **kwargs,
    )
    payload = {"protocol": manifest["protocol"], "jobs": len(manifest["jobs"]), "run_dir": str(run_dir.resolve())}
    if args.execute:
        results = execute_protocol_jobs(
            manifest,
            protocol_run_manifest_path=run_dir.resolve() / "protocol_run_manifest.json",
            gate=args.execute_gate,
            max_jobs=args.max_execute_jobs,
            data_root=args.data_root,
            device=args.device,
            epochs_override=args.execute_epochs_override,
            allow_artifact_only=args.allow_artifact_only,
        )
        executed = sum(1 for item in results if item.get("status") in {"EXECUTED", "EXECUTED_ARTIFACT"})
        payload["executed_jobs"] = executed
        payload["skipped_artifact_only_jobs"] = sum(1 for item in results if item.get("status") == "SKIPPED_ARTIFACT_ONLY")
        payload["failed_jobs"] = sum(
            1
            for item in results
            if item.get("status") == "FAILED_ARTIFACT"
            or item.get("status") == "SKIPPED_DEPENDENCY_FAILED"
            or (
                item.get("status") in {"EXECUTED", "EXECUTED_ARTIFACT"}
                and (item.get("command_returncode") != 0 or item.get("audit_returncode") not in {0, None})
            )
        )
        if payload["failed_jobs"]:
            print(json.dumps(payload, ensure_ascii=False))
            return 1
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
