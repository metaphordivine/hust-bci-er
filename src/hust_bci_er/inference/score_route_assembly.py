"""Assemble score-route prediction tables from component score tables."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from hust_bci_er.contracts.prediction import prediction_schema
from hust_bci_er.inference.clean_score_routes import score_route_by_id
from hust_bci_er.inference.score_fusion import assemble_score_route
from hust_bci_er.inference.topk import subject_top4


@dataclass(frozen=True)
class ComponentScoreTable:
    key_columns: tuple[str, ...]
    keys: tuple[tuple[str, ...], ...]
    scores: np.ndarray
    y_true: Mapping[tuple[str, ...], str]


def read_component_score_table(path: Path, *, component_id: str | None = None) -> ComponentScoreTable:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"component score table is empty: {path}")
    schema = prediction_schema(set(rows[0]))
    subject_col = schema["subject_id"]
    trial_col = schema["trial_id"]
    score_col = schema["score"]
    truth_col = schema["y_true"]
    if subject_col is None or trial_col is None or score_col is None:
        raise ValueError(f"component score table must include subject, trial, and score columns: {path}")
    if component_id is not None and "component_id" not in rows[0]:
        raise ValueError(f"component score table must include component_id for {component_id}: {path}")
    if component_id is not None:
        for idx, row in enumerate(rows):
            actual = row.get("component_id")
            if actual in {None, ""}:
                raise ValueError(f"component score table row {idx} component_id is empty for {component_id}: {path}")
            if str(actual) != component_id:
                raise ValueError(f"component score table component_id does not match {component_id}: {path} ({actual})")

    optional_key_cols: list[str] = []
    for col in ("seed", "fold", "crop_id"):
        if col not in rows[0]:
            continue
        present = [row.get(col) not in {None, ""} for row in rows]
        if any(present) and not all(present):
            raise ValueError(f"component score table has partial {col} values: {path}")
        if all(present):
            optional_key_cols.append(col)

    key_sources = tuple(optional_key_cols + [subject_col, trial_col])
    key_columns = tuple(optional_key_cols + ["subject_id", "trial_id"])
    keys: list[tuple[str, ...]] = []
    seen_keys: set[tuple[str, ...]] = set()
    scores: list[float] = []
    y_true: dict[tuple[str, ...], str] = {}
    for row in rows:
        key = tuple(str(row[col]) for col in key_sources)
        if key in seen_keys:
            raise ValueError(f"duplicate component alignment row in component score table: {path}")
        try:
            score = float(row[score_col])
        except ValueError as exc:
            raise ValueError(f"component score is not numeric in {path}: {key}") from exc
        if not np.isfinite(score):
            raise ValueError(f"component score is not finite in {path}: {key}")
        keys.append(key)
        seen_keys.add(key)
        scores.append(score)
        if truth_col is not None and row.get(truth_col) not in {None, ""}:
            try:
                truth = int(row[truth_col])
            except ValueError as exc:
                raise ValueError(f"component score y_true is not binary in {path}: {key}") from exc
            if truth not in {0, 1}:
                raise ValueError(f"component score y_true is not binary in {path}: {key}")
            y_true[key] = str(truth)
    return ComponentScoreTable(key_columns, tuple(keys), np.asarray(scores, dtype=np.float64), y_true)


def alignment_key(key_columns: tuple[str, ...], group_columns: tuple[str, ...], group: tuple[str, ...], trial: str) -> tuple[str, ...]:
    values = dict(zip(group_columns, group))
    values["trial_id"] = trial
    return tuple(values[col] for col in key_columns)


def matrices_from_component_tables(
    tables: Mapping[str, ComponentScoreTable],
    required_components: tuple[str, ...],
) -> tuple[tuple[str, ...], list[tuple[str, ...]], list[list[str]], dict[str, np.ndarray], dict[tuple[str, ...], str]]:
    first = tables[required_components[0]]
    first_keys = list(first.keys)
    if "trial_id" not in first.key_columns or "subject_id" not in first.key_columns:
        raise ValueError("component score table key columns must include subject_id and trial_id")
    trial_idx = first.key_columns.index("trial_id")
    group_indices = tuple(idx for idx, col in enumerate(first.key_columns) if col != "trial_id")
    group_columns = tuple(first.key_columns[idx] for idx in group_indices)

    groups: list[tuple[str, ...]] = []
    trials_by_group: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for key in first_keys:
        group_key = tuple(key[idx] for idx in group_indices)
        trial = key[trial_idx]
        if group_key not in trials_by_group:
            groups.append(group_key)
        trials_by_group[group_key].append(trial)

    trial_counts = {len(trials) for trials in trials_by_group.values()}
    if len(trial_counts) != 1:
        raise ValueError("all component alignment groups must have the same number of trial rows")
    n_trials = trial_counts.pop()

    component_matrices: dict[str, np.ndarray] = {}
    truth_by_key: dict[tuple[str, ...], str] = {}
    has_truth = any(bool(table.y_true) for table in tables.values())
    first_key_set = set(first_keys)
    for name in required_components:
        table = tables[name]
        if table.key_columns != first.key_columns:
            raise ValueError(f"component score table key columns do not match first component: {name}")
        if set(table.keys) != first_key_set:
            raise ValueError(f"component score table keys do not match first component: {name}")
        score_map = dict(zip(table.keys, table.scores))
        component_matrices[name] = np.asarray(
            [[score_map[alignment_key(first.key_columns, group_columns, group, trial)] for trial in trials_by_group[group]] for group in groups],
            dtype=np.float64,
        ).reshape(len(groups), n_trials)
        if has_truth:
            for key in first_keys:
                truth = table.y_true.get(key)
                if truth is None:
                    raise ValueError(f"component score table is missing y_true for alignment key: {name} {key}")
                existing = truth_by_key.get(key)
                if existing is not None and existing != truth:
                    raise ValueError(f"component score y_true mismatch for alignment key: {key}")
                truth_by_key[key] = truth

    return group_columns, groups, [trials_by_group[group] for group in groups], component_matrices, truth_by_key


def assemble_score_route_rows(route_config_path: Path, component_score_paths: Mapping[str, Path]) -> list[dict[str, str]]:
    data = yaml.safe_load(route_config_path.read_text(encoding="utf-8")) or {}
    route_id = data.get("route_id")
    route = score_route_by_id(str(route_id))
    if route is None:
        raise ValueError(f"route is not a registered score route: {route_id}")

    missing = [name for name in route.components if name not in component_score_paths]
    if missing:
        raise ValueError(f"missing component score tables: {missing}")

    tables = {name: read_component_score_table(Path(component_score_paths[name]), component_id=name) for name in route.components}
    group_columns, groups, trials_by_group, component_matrices, truth_by_key = matrices_from_component_tables(tables, route.components)
    route_scores = assemble_score_route(route, component_matrices)

    rows: list[dict[str, str]] = []
    for group_idx, group in enumerate(groups):
        group_values = dict(zip(group_columns, group))
        scores = route_scores[group_idx]
        if scores.size != 8:
            context = "|".join(group)
            raise ValueError(f"Top-4 score routes require 8 trial rows per alignment group: {context}")
        pred = subject_top4(scores)
        for trial, score, top4 in zip(trials_by_group[group_idx], scores, pred):
            item = {"route_id": route.route_id}
            item.update(group_values)
            item.update(
                {
                    "trial_id": trial,
                    "score": f"{float(score):.12g}",
                    "pred_top4": str(int(top4)),
                }
            )
            if truth_by_key:
                item["y_true"] = truth_by_key[alignment_key(tables[route.components[0]].key_columns, group_columns, group, trial)]
            rows.append(item)
    return rows


def write_score_route_rows(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_fields = [field for field in ("seed", "fold") if any(field in row for row in rows)]
    crop_fields = ["crop_id"] if any("crop_id" in row for row in rows) else []
    truth_fields = ["y_true"] if any("y_true" in row for row in rows) else []
    fieldnames = ["route_id", *metadata_fields, "subject_id", "trial_id", *crop_fields, "score", "pred_top4", *truth_fields]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
