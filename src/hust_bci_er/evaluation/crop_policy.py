"""Crop policy utilities for pseudo-public and stress-test evaluation."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

FIXED_CROP_POLICIES = {"crop1": 0, "crop2": 1, "crop3": 2, "crop4": 3, "crop5": 4}


def crop_policy_manifest(
    policy: str,
    *,
    seed: int,
    metric: str = "exact_single_crop_expected_BA",
    tie_break: str | None = None,
) -> dict[str, object]:
    """Return the manifest contract for a crop policy.

    P2 uses this to make random and worst-crop runs reproducible. ``worst`` is
    label-aware and is only valid for stress-test evaluation, not inference.
    """
    if policy in FIXED_CROP_POLICIES:
        return {
            "name": policy,
            "selection": "fixed_index",
            "crop_index": FIXED_CROP_POLICIES[policy],
            "tie_break": "not_applicable",
        }
    if policy == "single":
        return {
            "name": policy,
            "selection": "route_default_single_crop",
            "crop_index": 0,
            "tie_break": "not_applicable",
        }
    if policy == "exact_single_crop":
        return {
            "name": policy,
            "selection": "route_defined",
            "tie_break": tie_break or "stable_input_order",
        }
    if policy == "sliding_window_vote":
        return {
            "name": policy,
            "selection": "route_defined",
            "tie_break": tie_break or "mean_score",
        }
    if policy == "random":
        return {
            "name": policy,
            "selection": "per_trial_uniform_crop",
            "random_seed": int(seed),
            "rng": "numpy.default_rng",
            "tie_break": "stable_input_order",
        }
    if policy == "worst":
        return {
            "name": policy,
            "selection": "label_aware_min_metric_stress_test",
            "metric": metric,
            "tie_break": "lowest_assignment_index",
        }
    raise ValueError(f"unsupported crop policy: {policy}")


def route_crop_policy_manifest(route_data: Mapping[str, Any], *, seed: int) -> dict[str, object]:
    """Return the crop-policy manifest implied by a route config."""
    inference = route_data.get("inference") if isinstance(route_data, Mapping) else None
    policy = str(inference.get("crop_policy", "single")) if isinstance(inference, Mapping) else "single"
    augmentation = route_data.get("augmentation") if isinstance(route_data, Mapping) else None
    aggregate = augmentation.get("aggregate_to_trial") if isinstance(augmentation, Mapping) else None
    tie_break = aggregate.get("tie_break") if isinstance(aggregate, Mapping) else None
    return crop_policy_manifest(policy, seed=seed, tie_break=str(tie_break) if tie_break else None)


def select_crop_matrix(mat: np.ndarray, policy: str, random_state: int | None = None) -> np.ndarray:
    """Return a [trials, 1] matrix for fixed or random crop policies.

    `worst` is label-aware and intentionally not implemented here. It belongs
    in stress-test evaluation code, not real inference.
    """
    mat = np.asarray(mat, dtype=np.float64)
    if policy in FIXED_CROP_POLICIES:
        idx = FIXED_CROP_POLICIES[policy]
        if idx >= mat.shape[1]:
            raise ValueError(f"{policy} is out of range for {mat.shape[1]} crops")
        return mat[:, idx : idx + 1]
    if policy == "random":
        if random_state is None:
            raise ValueError("random crop policy requires random_state")
        rng = np.random.default_rng(random_state)
        idx = rng.integers(0, mat.shape[1], size=mat.shape[0])
        return mat[np.arange(mat.shape[0]), idx][:, None]
    raise ValueError(f"unsupported non-label crop policy: {policy}")


def worst_crop_score(scores_by_assignment: np.ndarray, y_true: np.ndarray, metric_fn) -> tuple[float, int]:
    """Return worst assignment score and index for a label-aware stress test."""
    values = np.asarray([metric_fn(row, y_true) for row in scores_by_assignment], dtype=np.float64)
    idx = int(np.argmin(values))
    return float(values[idx]), idx
