from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.evaluation.protocols import (
    build_protocol1_plan,
    build_protocol2_plan,
    build_protocol3_plan,
)
from hust_bci_er.evaluation.protocols.params import load_param_grid


def route_ids_from_files(paths: list[Path]) -> list[str]:
    import yaml

    route_ids: list[str] = []
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        route_id = data.get("route_id")
        if not isinstance(route_id, str) or not route_id:
            raise ValueError(f"route config has no route_id: {path}")
        route_ids.append(route_id)
    return route_ids


def parse_grid_size(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("grid size must use route_id=size")
    route_id, raw_size = value.split("=", 1)
    route_id = route_id.strip()
    try:
        size = int(raw_size)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid size must be an integer") from exc
    if not route_id or size <= 0:
        raise argparse.ArgumentTypeError("route_id must be non-empty and size must be positive")
    return route_id, size


def parse_route_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must use route_id=path")
    route_id, raw_path = value.split("=", 1)
    route_id = route_id.strip()
    path = Path(raw_path.strip())
    if not route_id or not str(path):
        raise argparse.ArgumentTypeError("route_id and path must be non-empty")
    return route_id, path


def load_param_grid_sizes(values: list[tuple[str, Path]]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for route_id, path in values:
        sizes[route_id] = len(load_param_grid(path))
    return sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run an evaluation protocol plan without training.")
    parser.add_argument("--protocol", choices=["p1", "p2", "p3"], required=True)
    parser.add_argument("--route", action="append", default=[], help="Route id. Can be repeated.")
    parser.add_argument("--route-config", action="append", type=Path, default=[], help="Route config path. Can be repeated.")
    parser.add_argument("--seed", action="append", type=int, default=[], help="P1 seed. Can be repeated.")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-holdout-subjects", type=int, default=12)
    parser.add_argument("--crop-policy", action="append", default=[], help="P2 crop policy. Can be repeated.")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--grid-size", action="append", type=parse_grid_size, default=[], help="P3 route grid size, route_id=size.")
    parser.add_argument("--param-grid", action="append", type=parse_route_path, default=[], help="P3 route parameter grid YAML, route_id=path. Overrides --grid-size count for that route.")
    args = parser.parse_args(argv)

    routes = list(args.route)
    routes.extend(route_ids_from_files(args.route_config))

    if args.protocol == "p1":
        plan = build_protocol1_plan(routes, seeds=args.seed or (42, 123, 456, 789, 1024), n_folds=args.n_folds)
    elif args.protocol == "p2":
        plan = build_protocol2_plan(routes, n_holdout_subjects=args.n_holdout_subjects, crop_policies=args.crop_policy or ("crop1", "crop2", "crop3", "crop4", "crop5", "random", "worst"))
    else:
        grid_sizes = dict(args.grid_size)
        grid_sizes.update(load_param_grid_sizes(args.param_grid))
        plan = build_protocol3_plan(routes, outer_folds=args.outer_folds, inner_folds=args.inner_folds, grid_sizes=grid_sizes)

    print(json.dumps(plan.as_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
