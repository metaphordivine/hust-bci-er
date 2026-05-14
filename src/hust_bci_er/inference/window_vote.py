"""Aggregate per-window class scores back to trial rows."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def strict_majority_class(
    classes: Iterable[int],
    *,
    scores: Iterable[float] | None = None,
    threshold: float = 0.5,
    tie_break: str = "mean_score",
) -> int:
    values = [int(value) for value in classes]
    if not values:
        raise ValueError("at least one class value is required")

    counts = Counter(values)
    best_count = max(counts.values())
    winners = sorted(value for value, count in counts.items() if count == best_count)
    if len(winners) == 1 and best_count > len(values) / 2:
        return winners[0]

    if tie_break == "mean_score":
        if scores is None:
            raise ValueError("mean_score tie break requires scores")
        mean_score = float(np.mean([float(score) for score in scores]))
        return int(mean_score >= threshold)
    if tie_break == "lower":
        return winners[0]
    if tie_break == "higher":
        return winners[-1]
    raise ValueError(f"unsupported tie_break: {tie_break}")


def aggregate_window_predictions(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_keys: tuple[str, ...] = ("subject_id", "trial_id"),
    score_key: str = "y_score",
    class_key: str = "y_pred",
    threshold: float = 0.5,
    tie_break: str = "mean_score",
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    out: list[dict[str, Any]] = []
    for key, group in grouped.items():
        has_score = [score_key in row and row.get(score_key) not in {None, ""} for row in group]
        has_class = [class_key in row and row.get(class_key) not in {None, ""} for row in group]
        if any(has_score) and not all(has_score):
            raise ValueError(f"all rows in group {key} must include {score_key}")
        if any(has_class) and not all(has_class):
            raise ValueError(f"all rows in group {key} must include {class_key}")
        scores = [float(row[score_key]) for row in group] if all(has_score) else []
        if all(has_class):
            classes = [int(row[class_key]) for row in group]
        elif all(has_score):
            classes = [int(score >= threshold) for score in scores]
        else:
            raise ValueError("rows must provide class values or scores")

        item = {field: value for field, value in zip(group_keys, key)}
        if scores:
            item[score_key] = float(np.mean(scores))
        item[class_key] = strict_majority_class(classes, scores=scores or None, threshold=threshold, tie_break=tie_break)
        item["n_windows"] = len(group)
        out.append(item)
    return out
