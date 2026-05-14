"""Contracts for reusable component artifacts."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from hust_bci_er.audit.manifest import sha256_file


COMPONENT_SCORE_COLUMNS = ["component_id", "subject_id", "trial_id", "crop_id", "score"]


@dataclass(frozen=True)
class ComponentScoreRecord:
    component_id: str
    subject_id: str
    trial_id: str
    crop_id: int | None
    score: float
    y_true: int | None = None
    fold: int | None = None
    seed: int | None = None


def write_component_scores(records: Iterable[ComponentScoreRecord], path: Path) -> dict[str, Any]:
    rows = [asdict(record) for record in records]
    errors = validate_component_score_rows(rows)
    if errors:
        raise ValueError("; ".join(errors))
    fieldnames = ["component_id", "subject_id", "trial_id", "crop_id", "score", "y_true", "fold", "seed"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"path": path.as_posix(), "sha256": sha256_file(path), "rows": len(rows), "schema": "component_scores_v1"}


def validate_component_score_rows(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen_keys: set[tuple[Any, ...]] = set()
    for idx, row in enumerate(rows):
        for col in COMPONENT_SCORE_COLUMNS:
            if col not in row:
                errors.append(f"row {idx} missing {col}")
            elif col != "crop_id" and row.get(col) in {None, ""}:
                errors.append(f"row {idx} {col} is empty")
        try:
            score = float(row.get("score"))
        except (TypeError, ValueError):
            errors.append(f"row {idx} score is not numeric")
        else:
            if not math.isfinite(score):
                errors.append(f"row {idx} score is not finite")
        if row.get("y_true") not in {None, ""}:
            try:
                y_true = int(row["y_true"])
            except (TypeError, ValueError):
                errors.append(f"row {idx} y_true is not binary")
            else:
                if y_true not in {0, 1}:
                    errors.append(f"row {idx} y_true is not binary")
        for col in ("fold", "seed"):
            if row.get(col) not in {None, ""}:
                try:
                    int(row[col])
                except (TypeError, ValueError):
                    errors.append(f"row {idx} {col} is not an integer")
        key = tuple(row.get(col) for col in ("component_id", "subject_id", "trial_id", "crop_id", "fold", "seed"))
        if key in seen_keys:
            errors.append(f"row {idx} duplicates component/subject/trial/crop/fold/seed key")
        seen_keys.add(key)
    return errors
