"""Small route schema validator.

This intentionally avoids a heavy validation framework for the initial skeleton.
It can be replaced by Pydantic later without changing the command contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import registry
from hust_bci_er.inference.clean_score_routes import score_route_by_id


REQUIRED_TOP_LEVEL = {
    "route_id",
    "status",
    "dataset_version",
    "split_id",
    "seed",
    "input_window_sec",
    "model",
    "inference",
    "evaluation",
}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def model_name(model: Any) -> str | None:
    if isinstance(model, str):
        return model
    if isinstance(model, dict):
        name = model.get("name")
        return str(name) if name is not None else None
    return None


def nested_name(value: Any, key: str = "name") -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        item = value.get(key)
        return str(item) if item is not None else None
    return None


def validate_route_config(data: dict[str, Any], path: Path | None = None) -> list[str]:
    errors: list[str] = []

    missing = sorted(REQUIRED_TOP_LEVEL.difference(data))
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")

    route_id = data.get("route_id")
    if not isinstance(route_id, str) or not route_id:
        errors.append("route_id must be a non-empty string")
    elif path is not None and path.suffix in {".yaml", ".yml"} and path.stem != route_id:
        errors.append(f"route_id must match file name: {path.stem}")

    status = data.get("status")
    if status not in registry.ROUTE_STATUSES:
        errors.append(f"invalid status: {status}")

    seed = data.get("seed")
    if not isinstance(seed, int):
        errors.append("seed must be an integer")

    input_window_sec = data.get("input_window_sec")
    if not isinstance(input_window_sec, (int, float)) or input_window_sec <= 0:
        errors.append("input_window_sec must be positive")

    for name in as_list(data.get("preprocessing")):
        if name not in registry.PREPROCESSING:
            errors.append(f"unknown preprocessing component: {name}")

    for name in as_list(data.get("features")):
        if name not in registry.FEATURES:
            errors.append(f"unknown feature component: {name}")

    name = model_name(data.get("model"))
    if name not in registry.MODELS:
        errors.append(f"unknown model component: {name}")
    model = data.get("model")
    if name == "score_fusion":
        validate_score_fusion_model(data, model, route_id, errors)
    elif isinstance(model, dict):
        for field in ["score_node", "components"]:
            if field in model:
                errors.append(f"model.{field} is only allowed for score_fusion routes")

    adaptation = nested_name(data.get("adaptation")) or "none"
    if adaptation not in registry.ADAPTATION:
        errors.append(f"unknown adaptation component: {adaptation}")

    evaluation = data.get("evaluation")
    protocol = None
    metric = None
    if isinstance(evaluation, dict):
        protocol = evaluation.get("protocol")
        metric = evaluation.get("primary_metric")
    else:
        errors.append("evaluation must be a mapping")

    if not protocol:
        errors.append("evaluation.protocol is required")
    if not metric:
        errors.append("evaluation.primary_metric is required")
    if protocol not in registry.EVALUATION_PROTOCOLS:
        errors.append(f"unknown evaluation protocol: {protocol}")
    if metric is not None and metric not in registry.PRIMARY_METRICS:
        errors.append(f"unknown primary metric: {metric}")

    inference = data.get("inference")
    top4 = None
    if not isinstance(inference, dict):
        errors.append("inference must be a mapping")
    else:
        top4 = inference.get("top4")
        if "top4" not in inference:
            errors.append("inference.top4 is required")
        if "crop_policy" not in inference:
            errors.append("inference.crop_policy is required")

    if metric == "top4_BA" and top4 is not True:
        errors.append("top4_BA requires inference.top4: true")
    if metric == "no_top4_BA" and top4 is not False:
        errors.append("no_top4_BA requires inference.top4: false")
    if metric in {"exact_single_crop_expected_BA", "all_correct_rate"} and top4 is not True:
        errors.append(f"{metric} requires inference.top4: true")

    for required in ["dataset_version", "split_id"]:
        if not data.get(required):
            errors.append(f"{required} must be set")

    dataset_version = data.get("dataset_version")
    if isinstance(dataset_version, str) and dataset_version:
        dataset_path = registry.ROOT / "configs" / "datasets" / f"{dataset_version}.yaml"
        if not dataset_path.exists():
            errors.append(f"dataset_version has no manifest: {dataset_version}")

    split_id = data.get("split_id")
    if isinstance(split_id, str) and split_id:
        split_path = registry.ROOT / "configs" / "splits" / f"{split_id}.yaml"
        if not split_path.exists():
            errors.append(f"split_id has no manifest: {split_id}")

    if "tier" in data:
        errors.append("tier is not allowed; all routes are peers before audit")
    if "output_dir" in data:
        errors.append("output_dir is derived from route_id/run_id and must not be set in route config")

    return errors


def validate_score_fusion_model(data: dict[str, Any], model: Any, route_id: Any, errors: list[str]) -> None:
    if not isinstance(model, dict):
        errors.append("score_fusion model must be a mapping with score_node and components")
        return

    score_node = model.get("score_node")
    if not isinstance(score_node, str) or not score_node:
        errors.append("score_fusion model.score_node is required")

    components = model.get("components")
    if not isinstance(components, list) or not components or not all(isinstance(item, str) and item for item in components):
        errors.append("score_fusion model.components must be a non-empty list of strings")
        return

    if not isinstance(route_id, str) or not route_id:
        return
    route = score_route_by_id(route_id)
    if route is None:
        errors.append(f"score_fusion route is not registered in CLEAN_SCORE_ROUTES: {route_id}")
        return
    if score_node != route.score_node:
        errors.append(f"score_fusion score_node must match registered node: {route.score_node}")
    if tuple(components) != route.components:
        errors.append(f"score_fusion components must match registered components: {list(route.components)}")
