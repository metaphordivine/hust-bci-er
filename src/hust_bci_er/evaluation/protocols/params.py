"""Parameter-grid helpers for protocol runner jobs."""

from __future__ import annotations

import copy
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


FORBIDDEN_OVERRIDE_KEYS = {
    "route_id",
    "status",
    "dataset_version",
    "split_id",
    "seed",
    "evaluation.protocol",
    "evaluation.primary_metric",
    "training.job_adapter",
    "training.trainer",
    "model.name",
}


def _deep_set(target: dict[str, Any], key_path: str, value: Any) -> None:
    keys = key_path.split(".")
    if not keys or any(not key for key in keys):
        raise ValueError(f"invalid parameter override path: {key_path!r}")
    for key in keys[:-1]:
        current = target.setdefault(key, {})
        if not isinstance(current, dict):
            raise ValueError(f"cannot set nested override under non-mapping path: {key_path}")
        target = current
    target[keys[-1]] = value


def validate_param_overrides(overrides: Mapping[str, Any]) -> None:
    for key in overrides:
        if not isinstance(key, str) or not key:
            raise ValueError("parameter override keys must be non-empty strings")
        if key in FORBIDDEN_OVERRIDE_KEYS:
            raise ValueError(f"parameter override cannot change protocol identity field: {key}")


def apply_param_overrides(route_data: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a route config with selected P3 parameter overrides applied."""
    validate_param_overrides(overrides)
    route = copy.deepcopy(dict(route_data))
    for key_path, value in sorted(overrides.items()):
        _deep_set(route, key_path, value)
    return route


def write_effective_route_config(
    *,
    base_route_data: Mapping[str, Any],
    overrides: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    """Write a run-local effective route YAML and return its path."""
    route = apply_param_overrides(base_route_data, overrides)
    route_id = str(route.get("route_id") or "selected_route")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{route_id}.yaml"
    path.write_text(yaml.safe_dump(route, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _values_from_spec(spec: Any) -> list[Any]:
    if isinstance(spec, list):
        return list(spec)
    if isinstance(spec, dict):
        for key in ("values", "coarse", "fine"):
            values = spec.get(key)
            if isinstance(values, list) and values:
                return list(values)
    return []


def param_grid_from_search_space(search_space: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand a search-space YAML payload into dot-path override combinations.

    The accepted shape matches ``scripts/hparam_search.py``:

    ``parameters.<dot.path>.values`` first, otherwise ``coarse``, otherwise
    ``fine``. This is a concrete one-stage candidate list, not the full
    coarse-to-fine search process from ``scripts/hparam_search.py``.
    """
    parameters = search_space.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("parameter grid search space must have a non-empty 'parameters' mapping")
    keys: list[str] = []
    value_lists: list[list[Any]] = []
    for key, spec in parameters.items():
        values = _values_from_spec(spec)
        if not values:
            raise ValueError(f"parameter grid has no values for {key}")
        keys.append(str(key))
        value_lists.append(values)
    combos: list[dict[str, Any]] = []
    for values in product(*value_lists):
        combo = dict(zip(keys, values))
        validate_param_overrides(combo)
        combos.append(combo)
    return combos


def load_param_grid(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"parameter grid file must be a mapping: {path}")
    return param_grid_from_search_space(payload)


def normalize_param_grids(
    param_grids: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> dict[str, tuple[dict[str, Any], ...]]:
    normalized: dict[str, tuple[dict[str, Any], ...]] = {}
    for route_id, candidates in (param_grids or {}).items():
        route_candidates = tuple(dict(candidate) for candidate in candidates)
        if not route_candidates:
            raise ValueError(f"P3 parameter grid is empty for route: {route_id}")
        for candidate in route_candidates:
            validate_param_overrides(candidate)
        normalized[str(route_id)] = route_candidates
    return normalized
