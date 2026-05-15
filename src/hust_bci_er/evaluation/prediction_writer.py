"""Prediction CSV writer for audited experiment artifacts."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from hust_bci_er.audit.manifest import sha256_file
from hust_bci_er.contracts.records import PredictionRecord
from hust_bci_er.inference.topk import topk_binary


PREDICTION_FIELDNAMES = ["route_id", "subject_id", "trial_id", "crop_id", "y_score", "y_pred", "y_true", "pred_top4", "fold", "seed"]


def rows_with_top4(
    records: Iterable[PredictionRecord],
    *,
    group_keys: tuple[str, ...] = ("subject_id",),
    group_size: int = 8,
    top_k: int = 4,
) -> list[dict]:
    rows = [asdict(record) for record in records]
    groups: dict[tuple, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[tuple(row.get(key) for key in group_keys)].append(idx)
    for group_key, indices in groups.items():
        if len(indices) != group_size:
            raise ValueError(f"Top-{top_k} prediction group {group_key} must contain {group_size} rows")
        scores = [float(rows[idx]["y_score"]) for idx in indices]
        top4 = topk_binary(scores, top_k)
        for local_idx, row_idx in enumerate(indices):
            rows[row_idx]["pred_top4"] = int(top4[local_idx])
    return rows


def write_predictions(
    records: Iterable[PredictionRecord],
    path: Path,
    *,
    group_keys: tuple[str, ...] = ("subject_id",),
    group_size: int = 8,
    top_k: int = 4,
    include_top4: bool = True,
) -> dict:
    if include_top4:
        rows = rows_with_top4(records, group_keys=group_keys, group_size=group_size, top_k=top_k)
    else:
        rows = [asdict(record) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return {"path": path.as_posix(), "sha256": sha256_file(path), "rows": len(rows), "schema": "prediction_v1"}
