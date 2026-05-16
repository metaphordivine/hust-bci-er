"""Registry consistency checks used by tests and repo doctor."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import yaml

from hust_bci_er.config import registry
from hust_bci_er.config.docs import render_component_docs
from hust_bci_er.inference.clean_score_routes import CLEAN_SCORE_ROUTES, score_route_by_id


PREPROCESSING_EVIDENCE = {
    "zscore": ("preprocessing/normalization.py", "zscore_per_channel"),
    "robust_zscore": ("preprocessing/normalization.py", "robust_zscore_per_channel"),
    "car": ("preprocessing/normalization.py", "common_average_reference"),
    "bandpass": ("preprocessing/filtering.py", "bandpass_filter"),
    "euclidean_alignment": ("preprocessing/euclidean_alignment.py", "fit_ea_transform"),
    "whitening_eps1e3": ("preprocessing/whitening.py", "channel_whiten"),
    "whitening_eps3e4": ("preprocessing/whitening.py", "channel_whiten"),
    "shrinkage_whitening": ("preprocessing/whitening.py", "shrinkage_alpha"),
}
FEATURE_EVIDENCE = {
    "bandpower": ("features/bandpower.py", "bandpower_features"),
    "connectivity": ("features/connectivity.py", "connectivity_features"),
    "differential_entropy": ("features/differential_entropy.py", "differential_entropy_features"),
    "hjorth": ("features/hjorth.py", "hjorth_features"),
    "riemannian_tangent": ("features/riemannian.py", "tangent_space_features"),
}


def model_factory_keys(root: Path) -> set[str]:
    tree = ast.parse((root / "src" / "hust_bci_er" / "models" / "factory.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "BUILDERS" for target in node.targets):
            if isinstance(node.value, ast.Dict):
                return {key.value for key in node.value.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    raise ValueError("BUILDERS mapping not found")


def graph_factory_keys(root: Path) -> set[str]:
    tree = ast.parse((root / "src" / "hust_bci_er" / "models" / "factory.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "GRAPH_BUILDERS" for target in node.targets):
            if isinstance(node.value, ast.Dict):
                return {key.value for key in node.value.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    raise ValueError("GRAPH_BUILDERS mapping not found")


def route_component_names(root: Path) -> dict[str, set[str]]:
    out = {"models": set(), "preprocessing": set(), "features": set(), "score_routes": set()}
    for path in (root / "configs" / "routes" / "models").glob("*.yaml"):
        data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        model = data.get("model")
        if isinstance(model, dict) and model.get("name"):
            out["models"].add(str(model["name"]))
            if model.get("name") == "score_fusion" and data.get("route_id"):
                out["score_routes"].add(str(data["route_id"]))
        for item in data.get("preprocessing") or []:
            if isinstance(item, dict):
                out["preprocessing"].add(str(item.get("name")))
            else:
                out["preprocessing"].add(str(item))
        for item in data.get("features") or []:
            out["features"].add(str(item))
    return out


def file_contains(root: Path, rel_path: str, token: str) -> bool:
    path = root / "src" / "hust_bci_er" / rel_path
    return path.exists() and token in path.read_text(encoding="utf-8", errors="ignore")


def registry_consistency_errors(root: Path) -> list[str]:
    errors: list[str] = []
    if model_factory_keys(root) != registry.TORCH_BACKBONES:
        errors.append("TORCH_BACKBONES must match models.factory BUILDERS")
    if graph_factory_keys(root) != registry.GRAPH_MODELS:
        errors.append("GRAPH_MODELS must match models.factory GRAPH_BUILDERS")
    used = route_component_names(root)
    unknown_models = sorted(used["models"] - registry.MODELS)
    unknown_preprocessing = sorted(used["preprocessing"] - registry.PREPROCESSING)
    unknown_features = sorted(used["features"] - registry.FEATURES)
    if unknown_models:
        errors.append("route configs use unknown models: " + ", ".join(unknown_models))
    if unknown_preprocessing:
        errors.append("route configs use unknown preprocessing: " + ", ".join(unknown_preprocessing))
    if unknown_features:
        errors.append("route configs use unknown features: " + ", ".join(unknown_features))
    for name in sorted(used["preprocessing"]):
        evidence = PREPROCESSING_EVIDENCE.get(name)
        if evidence is not None and not file_contains(root, *evidence):
            errors.append(f"preprocessing registry item lacks implementation evidence: {name}")
    for name in sorted(used["features"]):
        evidence = FEATURE_EVIDENCE.get(name)
        if evidence is not None and not file_contains(root, *evidence):
            errors.append(f"feature registry item lacks implementation evidence: {name}")
    registered_score_routes = {route.route_id for route in CLEAN_SCORE_ROUTES}
    unknown_score_routes = sorted(used["score_routes"] - registered_score_routes)
    if unknown_score_routes:
        errors.append("score_fusion route configs are not registered: " + ", ".join(unknown_score_routes))
    for path in (root / "configs" / "routes" / "models").glob("*.yaml"):
        data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        model = data.get("model")
        if not isinstance(model, dict) or model.get("name") != "score_fusion":
            continue
        route = score_route_by_id(str(data.get("route_id")))
        if route is None:
            continue
        if model.get("score_node") != route.score_node:
            errors.append(f"{path.name} score_node does not match clean score route registry")
        if tuple(model.get("components") or ()) != route.components:
            errors.append(f"{path.name} components do not match clean score route registry")
    docs_path = root / "docs" / "component_registry.md"
    if docs_path.exists() and docs_path.read_text(encoding="utf-8") != render_component_docs():
        errors.append("docs/component_registry.md is not synchronized with registry")
    return errors
