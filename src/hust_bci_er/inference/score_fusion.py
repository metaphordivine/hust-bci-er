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
    raw_weights = [1.0 / len(arrays)] * len(arrays) if weights is None else weights
    if len(raw_weights) != len(arrays):
        raise ValueError("weights length must match arrays")
    normalized_weights = np.asarray(raw_weights, dtype=np.float64)
    if not np.isfinite(normalized_weights).all():
        raise ValueError("weights must be finite")
    if np.any(normalized_weights < 0):
        raise ValueError("weights must be non-negative")
    weight_sum = float(normalized_weights.sum())
    if weight_sum <= 0:
        raise ValueError("weights sum must be positive")
    normalized_weights = normalized_weights / weight_sum
    out = np.zeros_like(np.asarray(arrays[0], dtype=np.float64))
    for weight, arr in zip(normalized_weights, arrays):
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
    if route.method == "margin_adaptive_query_context":
        if route.query_component is None or not route.context_components:
            raise ValueError(f"margin_adaptive_query_context route is missing query/context components: {route.route_id}")
        query = component_matrix(route.query_component, component_scores, first.shape)
        context = [component_matrix(name, component_scores, first.shape) for name in route.context_components]
        alpha = 0.25 if route.alpha is None else float(route.alpha)
        alpha_max = 0.45 if route.adaptive_alpha_max is None else float(route.adaptive_alpha_max)
        temperature = 1.0 if route.temperature is None else float(route.temperature)
        margin_low = 0.15 if route.margin_low is None else float(route.margin_low)
        margin_high = 0.85 if route.margin_high is None else float(route.margin_high)
        return margin_adaptive_query_context_fusion(
            query,
            context,
            alpha=alpha,
            alpha_max=alpha_max,
            temperature=temperature,
            margin_low=margin_low,
            margin_high=margin_high,
        )
    if route.method == "hardness_router_query_context":
        if route.query_component is None or not route.context_components:
            raise ValueError(f"hardness_router_query_context route is missing query/context components: {route.route_id}")
        query = component_matrix(route.query_component, component_scores, first.shape)
        context = [component_matrix(name, component_scores, first.shape) for name in route.context_components]
        alpha = 0.15 if route.alpha is None else float(route.alpha)
        alpha_max = 0.40 if route.adaptive_alpha_max is None else float(route.adaptive_alpha_max)
        temperature = 1.0 if route.temperature is None else float(route.temperature)
        margin_low = 0.15 if route.margin_low is None else float(route.margin_low)
        margin_high = 0.85 if route.margin_high is None else float(route.margin_high)
        return hardness_router_query_context_fusion(
            query,
            context,
            alpha=alpha,
            alpha_max=alpha_max,
            temperature=temperature,
            margin_low=margin_low,
            margin_high=margin_high,
        )
    raise ValueError(f"unknown score route method: {route.method}")


def assemble_score_node(score_node: str, component_scores: Mapping[str, np.ndarray]) -> np.ndarray:
    route = score_route_by_node(score_node)
    if route is None:
        raise KeyError(f"unknown score node: {score_node}")
    return assemble_score_route(route, component_scores)


def query_context_fusion(query: np.ndarray, context_scores: list[np.ndarray], *, alpha: float, temperature: float) -> np.ndarray:
    q = zscore_rows(query)
    context = query_context_matrix(q, context_scores, temperature=temperature)
    return (1.0 - float(alpha)) * q + float(alpha) * context


def query_context_matrix(query_zscores: np.ndarray, context_scores: list[np.ndarray], *, temperature: float) -> np.ndarray:
    if not context_scores:
        raise ValueError("at least one context score array is required")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    q = np.asarray(query_zscores, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError("query score array must be two-dimensional")
    context_items = [zscore_rows(v) for v in context_scores]
    for item in context_items:
        if item.shape != q.shape:
            raise ValueError(f"context score shape mismatch: expected {q.shape}, got {item.shape}")
    sims = np.stack([(q * v).mean(axis=1) for v in context_items], axis=1) / float(temperature)
    sims = sims - sims.max(axis=1, keepdims=True)
    weights = np.exp(sims)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return sum(weights[:, i : i + 1] * context_items[i] for i in range(len(context_items)))


def top4_boundary_margin(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("score arrays must be two-dimensional")
    if x.shape[1] < 5:
        raise ValueError("top-4 boundary margin requires at least 5 trial scores")
    ordered = np.sort(x, axis=1)[:, ::-1]
    return ordered[:, 3] - ordered[:, 4]


def margin_adaptive_query_context_fusion(
    query: np.ndarray,
    context_scores: list[np.ndarray],
    *,
    alpha: float,
    alpha_max: float,
    temperature: float,
    margin_low: float,
    margin_high: float,
) -> np.ndarray:
    if not (0.0 <= alpha <= alpha_max <= 1.0):
        raise ValueError("adaptive alpha must satisfy 0 <= alpha <= alpha_max <= 1")
    if margin_high <= margin_low:
        raise ValueError("margin_high must be greater than margin_low")
    q = zscore_rows(query)
    context = query_context_matrix(q, context_scores, temperature=temperature)
    margin = top4_boundary_margin(q)
    uncertainty = np.clip((float(margin_high) - margin) / (float(margin_high) - float(margin_low)), 0.0, 1.0)
    adaptive_alpha = float(alpha) + uncertainty * (float(alpha_max) - float(alpha))
    return (1.0 - adaptive_alpha[:, None]) * q + adaptive_alpha[:, None] * context


def component_disagreement(query_zscores: np.ndarray, context_zscores: list[np.ndarray]) -> np.ndarray:
    q = np.asarray(query_zscores, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError("query score array must be two-dimensional")
    if not context_zscores:
        raise ValueError("at least one context score array is required")
    items = [q] + [np.asarray(v, dtype=np.float64) for v in context_zscores]
    for item in items:
        if item.shape != q.shape:
            raise ValueError(f"context score shape mismatch: expected {q.shape}, got {item.shape}")
    return np.stack(items, axis=0).std(axis=0).mean(axis=1)


def hardness_router_query_context_fusion(
    query: np.ndarray,
    context_scores: list[np.ndarray],
    *,
    alpha: float,
    alpha_max: float,
    temperature: float,
    margin_low: float,
    margin_high: float,
) -> np.ndarray:
    if not (0.0 <= alpha <= alpha_max <= 1.0):
        raise ValueError("adaptive alpha must satisfy 0 <= alpha <= alpha_max <= 1")
    if margin_high <= margin_low:
        raise ValueError("margin_high must be greater than margin_low")
    q = zscore_rows(query)
    context_items = [zscore_rows(v) for v in context_scores]
    context = query_context_matrix(q, context_scores, temperature=temperature)

    margin = top4_boundary_margin(q)
    margin_uncertainty = np.clip((float(margin_high) - margin) / (float(margin_high) - float(margin_low)), 0.0, 1.0)
    disagreement = component_disagreement(q, context_items)
    disagreement_scale = disagreement / (disagreement + 1.0)
    hardness = np.maximum(margin_uncertainty, disagreement_scale)
    adaptive_alpha = float(alpha) + hardness * (float(alpha_max) - float(alpha))
    return (1.0 - adaptive_alpha[:, None]) * q + adaptive_alpha[:, None] * context
