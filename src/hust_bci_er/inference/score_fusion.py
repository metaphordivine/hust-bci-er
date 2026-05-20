"""Score-level fusion helpers for route assembly."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from hust_bci_er.inference.clean_score_routes import ScoreRoute, score_route_by_node


def zscore_rows(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("score arrays must be two-dimensional")
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / np.maximum(sd, eps)


def z_average(*arrays: np.ndarray, weights: list[float] | None = None) -> np.ndarray:
    if not arrays:
        raise ValueError("at least one score array is required")
    weights = weights or [1.0 / len(arrays)] * len(arrays)
    if len(weights) != len(arrays):
        raise ValueError("weights length must match arrays")
    out = np.zeros_like(np.asarray(arrays[0], dtype=np.float64))
    for weight, arr in zip(weights, arrays):
        out += float(weight) * zscore_rows(arr)
    return out


def softmax_rows(x: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = zscore_rows(x) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    probs = np.exp(z)
    return probs / probs.sum(axis=1, keepdims=True)


def calibrated_probability_average(*arrays: np.ndarray, weights: list[float] | None = None, temperature: float = 1.0) -> np.ndarray:
    if not arrays:
        raise ValueError("at least one score array is required")
    weights = weights or [1.0 / len(arrays)] * len(arrays)
    if len(weights) != len(arrays):
        raise ValueError("weights length must match arrays")
    out = np.zeros_like(np.asarray(arrays[0], dtype=np.float64))
    for weight, arr in zip(weights, arrays):
        out += float(weight) * softmax_rows(arr, temperature=temperature)
    return out


def component_matrix(name: str, component_scores: Mapping[str, np.ndarray], expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    if name not in component_scores:
        raise KeyError(f"missing component score: {name}")
    scores = np.asarray(component_scores[name], dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"component score must be two-dimensional: {name}")
    if not np.isfinite(scores).all():
        raise ValueError(f"component score contains non-finite values: {name}")
    if expected_shape is not None and scores.shape != expected_shape:
        raise ValueError(f"component score shape mismatch for {name}: expected {expected_shape}, got {scores.shape}")
    return scores


def assemble_score_route(route: ScoreRoute, component_scores: Mapping[str, np.ndarray]) -> np.ndarray:
    first = component_matrix(route.components[0], component_scores)
    arrays = [first] + [component_matrix(name, component_scores, first.shape) for name in route.components[1:]]

    if route.method == "z_average":
        weights = list(route.weights) if route.weights is not None else None
        return z_average(*arrays, weights=weights)
    if route.method == "calibrated_probability_average":
        weights = list(route.weights) if route.weights is not None else None
        temperature = 1.0 if route.temperature is None else float(route.temperature)
        return calibrated_probability_average(*arrays, weights=weights, temperature=temperature)
    if route.method == "query_context":
        if route.query_component is None or not route.context_components:
            raise ValueError(f"query_context route is missing query/context components: {route.route_id}")
        query = component_matrix(route.query_component, component_scores, first.shape)
        context = [component_matrix(name, component_scores, first.shape) for name in route.context_components]
        alpha = 0.25 if route.alpha is None else float(route.alpha)
        temperature = 1.0 if route.temperature is None else float(route.temperature)
        return query_context_fusion(query, context, alpha=alpha, temperature=temperature)
    raise ValueError(f"unknown score route method: {route.method}")


def assemble_score_node(score_node: str, component_scores: Mapping[str, np.ndarray]) -> np.ndarray:
    route = score_route_by_node(score_node)
    if route is None:
        raise KeyError(f"unknown score node: {score_node}")
    return assemble_score_route(route, component_scores)


def query_context_fusion(query: np.ndarray, context_scores: list[np.ndarray], *, alpha: float, temperature: float) -> np.ndarray:
    q = zscore_rows(query)
    context_items = [zscore_rows(v) for v in context_scores]
    sims = np.stack([(q * v).mean(axis=1) for v in context_items], axis=1) / float(temperature)
    sims = sims - sims.max(axis=1, keepdims=True)
    weights = np.exp(sims)
    weights = weights / weights.sum(axis=1, keepdims=True)
    context = sum(weights[:, i : i + 1] * context_items[i] for i in range(len(context_items)))
    return (1.0 - float(alpha)) * q + float(alpha) * context
