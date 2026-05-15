from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.training.real_adapter import run_real_classifier_route  # noqa: E402
from hust_bci_er.training.toy_adapter import run_toy_route  # noqa: E402


SUPPORTED_ADAPTERS = {
    "toy_centroid": run_toy_route,
    "torch_classifier": run_real_classifier_route,
}


def default_run_dir(route: Path) -> Path:
    return ROOT / "outputs" / route.stem / "latest"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a route through a stable training job adapter.")
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--split-id")
    parser.add_argument("--seed", type=int)
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
    adapter = SUPPORTED_ADAPTERS[adapter_name]
    adapter_kwargs: dict = dict(
        route_config_path=args.route,
        run_dir=run_dir,
        split_id=args.split_id,
        seed=args.seed,
        command=["python", "scripts/train_route.py", "--route", args.route.as_posix(), "--run-dir", run_dir.as_posix()],
    )
    if adapter_name == "torch_classifier":
        adapter_kwargs.update(
            smoke_epochs=args.smoke_epochs,
            smoke_n_dep=args.smoke_dep,
            smoke_n_hc=args.smoke_hc,
            device=args.device,
        )
    artifacts = adapter(**adapter_kwargs)
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

