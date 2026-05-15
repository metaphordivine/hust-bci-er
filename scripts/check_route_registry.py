from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "reports" / "route_registry.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML document must be a mapping: {path}")
    return data


def route_ids_from_configs() -> set[str]:
    route_ids: set[str] = set()
    for path in (ROOT / "configs" / "routes" / "models").glob("*.yaml"):
        data = load_yaml(path)
        route_id = data.get("route_id")
        if isinstance(route_id, str) and route_id:
            route_ids.add(route_id)
    return route_ids


def guarded_path_exists(value: str) -> bool:
    path = ROOT / value
    if value.endswith("/"):
        return path.is_dir()
    return path.exists()


def main() -> int:
    errors: list[str] = []
    if not REGISTRY.exists():
        errors.append("route registry is missing: reports/route_registry.yaml")
    else:
        data = load_yaml(REGISTRY)
        routes = data.get("routes")
        if not isinstance(routes, list):
            errors.append("route registry routes must be a list")
            routes = []
        entries = [item for item in routes if isinstance(item, dict)]
        if len(entries) != len(routes):
            errors.append("route registry routes must contain only mappings")
        listed = [str(item.get("route_id") or "") for item in entries]
        duplicate = sorted(route_id for route_id, count in Counter(listed).items() if route_id and count > 1)
        if duplicate:
            errors.append("route registry has duplicate route_id values: " + ", ".join(duplicate))
        expected = route_ids_from_configs()
        actual = {route_id for route_id in listed if route_id}
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            errors.append("route registry missing routes: " + ", ".join(missing))
        if extra:
            errors.append("route registry has unknown routes: " + ", ".join(extra))
        for item in entries:
            route_id = item.get("route_id")
            owner = item.get("owner")
            if not isinstance(route_id, str) or not route_id:
                errors.append("route registry entry missing route_id")
            if not isinstance(owner, str) or not owner.strip():
                errors.append(f"route registry entry missing owner: {route_id}")
        guarded = data.get("shared_guarded_paths")
        if not isinstance(guarded, list) or not guarded:
            errors.append("route registry shared_guarded_paths must be a non-empty list")
        else:
            for value in guarded:
                if not isinstance(value, str) or not value:
                    errors.append("shared_guarded_paths entries must be strings")
                elif not guarded_path_exists(value):
                    errors.append(f"shared guarded path does not exist: {value}")

    if errors:
        print("Route registry check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Route registry check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
