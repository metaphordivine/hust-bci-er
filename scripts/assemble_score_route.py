from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from hust_bci_er.config.schema import validate_route_config
from hust_bci_er.inference.score_route_assembly import assemble_score_route_rows, write_score_route_rows


def parse_component_score(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("component score must use component=path")
    component, path = value.split("=", 1)
    component = component.strip()
    if not component:
        raise argparse.ArgumentTypeError("component name is empty")
    return component, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble a score_fusion route from component score CSV files.")
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--component-score", action="append", type=parse_component_score, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    route_data = yaml.safe_load(args.route.read_text(encoding="utf-8")) or {}
    errors = validate_route_config(route_data, path=args.route)
    if errors:
        print(f"route config is invalid: {args.route}", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    component_scores = dict(args.component_score)
    rows = assemble_score_route_rows(args.route, component_scores)
    write_score_route_rows(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
