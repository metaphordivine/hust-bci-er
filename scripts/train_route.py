from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.training.toy_adapter import run_toy_route  # noqa: E402


SUPPORTED_ADAPTERS = {
    "toy_centroid": run_toy_route,
    "torch_classifier": None,  # lazy import to avoid h5py hard dependency for toy routes
}


def _resolve_adapter(name: str):
    adapter = SUPPORTED_ADAPTERS.get(name)
    if adapter is None and name == "torch_classifier":
        from hust_bci_er.training.real_adapter import run_real_classifier_route  # noqa: E402
        SUPPORTED_ADAPTERS["torch_classifier"] = run_real_classifier_route
        return run_real_classifier_route
    if adapter is None:
        raise ValueError(f"unknown adapter: {name}")
    return adapter


def default_run_dir(route: Path) -> Path:
    return ROOT / "outputs" / route.stem / "latest"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a route through a stable training job adapter.")
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--split-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=["smoke", "full_subjects", "candidate"], default="smoke")
    parser.add_argument("--epochs-override", type=int, help="Override route training epochs for bounded local validation.")
    parser.add_argument("--smoke-epochs", type=int)
    parser.add_argument("--smoke-dep", type=int, default=4)
    parser.add_argument("--smoke-hc", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    route_data = yaml.safe_load(args.route.read_text(encoding="utf-8")) or {}
    if not isinstance(route_data, dict):
        raise ValueError(f"route config must be a mapping: {args.route}")

    training = route_data.get("training")
    adapter_name = training.get("job_adapter") if isinstance(training, dict) else None
    if adapter_name not in SUPPORTED_ADAPTERS:
        supported = ", ".join(sorted(SUPPORTED_ADAPTERS))
        raise ValueError(
            f"route {route_data.get('route_id', args.route.stem)} has "
            f"training.job_adapter={adapter_name or 'missing'}, "
            f"supported adapters: {supported}"
        )

    run_dir = args.run_dir or default_run_dir(args.route)

    # Build full command string for manifest provenance
    cmd = ["python", "scripts/train_route.py", "--mode", args.mode, "--route", args.route.as_posix(), "--run-dir", run_dir.as_posix()]
    if args.split_id is not None:
        cmd.extend(["--split-id", args.split_id])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    epochs_override = args.epochs_override if args.epochs_override is not None else args.smoke_epochs
    if args.epochs_override is not None:
        cmd.extend(["--epochs-override", str(args.epochs_override)])
    elif args.smoke_epochs is not None:
        cmd.extend(["--smoke-epochs", str(args.smoke_epochs)])
    cmd.extend(["--smoke-dep", str(args.smoke_dep)])
    cmd.extend(["--smoke-hc", str(args.smoke_hc)])
    cmd.extend(["--device", args.device])

    adapter = _resolve_adapter(adapter_name)
    artifacts = adapter(
        route_config_path=args.route,
        run_dir=run_dir,
        run_mode=args.mode,
        split_id=args.split_id,
        seed=args.seed,
        command=cmd,
        smoke_epochs=epochs_override,
        smoke_n_dep=args.smoke_dep,
        smoke_n_hc=args.smoke_hc,
        device=args.device,
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

