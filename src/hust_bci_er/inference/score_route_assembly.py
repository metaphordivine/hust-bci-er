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
    keys: tuple[tuple[str, str], ...]
    scores: np.ndarray


def read_component_score_table(path: Path) -> ComponentScoreTable:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"component score table is empty: {path}")
    schema = prediction_schema(set(rows[0]))
    subject_col = schema["subject_id"]
    trial_col = schema["trial_id"]
    score_col = schema["score"]
    if subject_col is None or trial_col is None or score_col is None:
        raise ValueError(f"component score table must include subject, trial, and score columns: {path}")

    keys: list[tuple[str, str]] = []
    scores: list[float] = []
    for row in rows:
        key = (str(row[subject_col]), str(row[trial_col]))
        if key in keys:
            raise ValueError(f"duplicate subject/trial row in component score table: {path}")
        try:
            score = float(row[score_col])
        except ValueError as exc:
            raise ValueError(f"component score is not numeric in {path}: {key}") from exc
        if not np.isfinite(score):
            raise ValueError(f"component score is not finite in {path}: {key}")
        keys.append(key)
        scores.append(score)
    return ComponentScoreTable(tuple(keys), np.asarray(scores, dtype=np.float64))


def matrices_from_component_tables(tables: Mapping[str, ComponentScoreTable], required_components: tuple[str, ...]) -> tuple[list[str], list[list[str]], dict[str, np.ndarray]]:
    first = tables[required_components[0]]
    first_keys = list(first.keys)
    subjects: list[str] = []
    trials_by_subject: dict[str, list[str]] = defaultdict(list)
    for subject, trial in first_keys:
        if subject not in trials_by_subject:
            subjects.append(subject)
        trials_by_subject[subject].append(trial)

    trial_counts = {len(trials) for trials in trials_by_subject.values()}
    if len(trial_counts) != 1:
        raise ValueError("all subjects must have the same number of trial rows")
    n_trials = trial_counts.pop()

    component_matrices: dict[str, np.ndarray] = {}
    first_key_set = set(first_keys)
    for name in required_components:
        table = tables[name]
        if set(table.keys) != first_key_set:
            raise ValueError(f"component score table keys do not match first component: {name}")
        score_map = dict(zip(table.keys, table.scores))
        component_matrices[name] = np.asarray(
            [[score_map[(subject, trial)] for trial in trials_by_subject[subject]] for subject in subjects],
            dtype=np.float64,
        ).reshape(len(subjects), n_trials)

    return subjects, [trials_by_subject[subject] for subject in subjects], component_matrices


def assemble_score_route_rows(route_config_path: Path, component_score_paths: Mapping[str, Path]) -> list[dict[str, str]]:
    data = yaml.safe_load(route_config_path.read_text(encoding="utf-8")) or {}
    route_id = data.get("route_id")
    route = score_route_by_id(str(route_id))
    if route is None:
        raise ValueError(f"route is not a registered score route: {route_id}")

    missing = [name for name in route.components if name not in component_score_paths]
    if missing:
        raise ValueError(f"missing component score tables: {missing}")

    tables = {name: read_component_score_table(Path(component_score_paths[name])) for name in route.components}
    subjects, trials_by_subject, component_matrices = matrices_from_component_tables(tables, route.components)
    route_scores = assemble_score_route(route, component_matrices)

    rows: list[dict[str, str]] = []
    for subject_idx, subject in enumerate(subjects):
        scores = route_scores[subject_idx]
        if scores.size != 8:
            raise ValueError(f"Top-4 score routes require 8 trial rows per subject: {subject}")
        pred = subject_top4(scores)
        for trial, score, top4 in zip(trials_by_subject[subject_idx], scores, pred):
            rows.append(
                {
                    "route_id": route.route_id,
                    "subject_id": subject,
                    "trial_id": trial,
                    "score": f"{float(score):.12g}",
                    "pred_top4": str(int(top4)),
                }
            )
    return rows


def write_score_route_rows(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["route_id", "subject_id", "trial_id", "score", "pred_top4"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
