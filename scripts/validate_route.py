from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from hust_bci_er.config.schema import validate_route_config  # noqa: E402


def route_files() -> list[Path]:
    return sorted((ROOT / "configs" / "routes").glob("**/*.yaml"))


def validate_file(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return ["route config must be a mapping"]
    return validate_route_config(data, path=path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("route", nargs="?", type=Path)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)

    paths = route_files() if args.all else [args.route]
    paths = [p for p in paths if p is not None]
    if not paths:
        print("No route configs selected.", file=sys.stderr)
        return 2

    seen: dict[str, Path] = {}
    failures = 0
    for path in paths:
        errors = validate_file(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        route_id = data.get("route_id")
        if route_id in seen:
            errors.append(f"duplicate route_id also used by {seen[route_id]}")
        elif route_id:
            seen[route_id] = path

        if errors:
            failures += 1
            print(f"FAIL {path}")
            for err in errors:
                print(f"  - {err}")
        else:
            print(f"PASS {path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

