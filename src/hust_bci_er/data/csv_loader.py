"""Small CSV-backed dataset loader used by toy and CI experiment runs."""

from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from hust_bci_er.contracts.records import EEGTrial, EEGWindow
from hust_bci_er.data.batch_contract import ModelBatch, validate_model_input_contract
from hust_bci_er.data.loader_contract import TrialCrop, trial_crop_from_index_row


FEATURE_PREFIX = "feature_"


def read_feature_vector(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"feature CSV must contain exactly one row: {path}")
    row = rows[0]
    feature_names = tuple(name for name in (reader.fieldnames or []) if str(name).startswith(FEATURE_PREFIX))
    if not feature_names:
        raise ValueError(f"feature CSV has no {FEATURE_PREFIX} columns: {path}")
    values = []
    for name in feature_names:
        try:
            values.append(float(row[name]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"feature column {name} is not numeric in {path}") from exc
    return np.asarray(values, dtype=np.float32), feature_names


class CsvManifestLoader:
    """Load trial rows from an auditable dataset manifest.

    The loader intentionally returns ``x`` separately from metadata so runtime
    checks can prove subject/trial identifiers are not part of model features.
    """

    def __init__(self, manifest: Mapping[str, Any], *, base_dir: Path) -> None:
        rows = manifest.get("trial_index") or manifest.get("trial_rows") or []
        if not isinstance(rows, list) or not rows:
            raise ValueError("dataset manifest must include non-empty trial_index or trial_rows")
        self._rows = [row for row in rows if isinstance(row, Mapping)]
        if len(self._rows) != len(rows):
            raise ValueError("dataset manifest rows must be mappings")
        self.base_dir = base_dir

    def iter_crops(self) -> Iterator[TrialCrop]:
        for row in self._rows:
            crop = trial_crop_from_index_row(row)
            if crop.path is None:
                raise ValueError("trial row path is required for CsvManifestLoader")
            path = Path(crop.path)
            resolved = path if path.is_absolute() else self.base_dir / path
            x, _ = read_feature_vector(resolved)
            yield trial_crop_from_index_row(row, x=x)

    def iter_trials(self) -> Iterator[EEGTrial]:
        for crop in self.iter_crops():
            yield EEGTrial(x=crop.x, y=crop.y, subject_id=crop.subject_id, trial_id=crop.trial_id, split=crop.split or "")

    def iter_windows(self) -> Iterator[EEGWindow]:
        for crop in self.iter_crops():
            yield EEGWindow(
                x=crop.x,
                y=crop.y,
                subject_id=crop.subject_id,
                trial_id=crop.trial_id,
                crop_id=crop.crop_id,
                window_start_sec=None,
                split=crop.split or "",
            )

    def model_batch(self, *, split: str | None = None) -> ModelBatch:
        crops = [crop for crop in self.iter_crops() if split is None or crop.split == split]
        if not crops:
            raise ValueError(f"no rows found for split: {split}")
        x = np.stack([crop.x for crop in crops], axis=0)
        y_values = [crop.y for crop in crops]
        y = None if any(value is None for value in y_values) else np.asarray(y_values, dtype=np.int64)
        metadata = {
            "subject_id": [crop.subject_id for crop in crops],
            "trial_id": [crop.trial_id for crop in crops],
            "crop_id": [crop.crop_id for crop in crops],
            "split": [crop.split for crop in crops],
            "path": [crop.path for crop in crops],
        }
        feature_count = x.shape[-1]
        batch = ModelBatch(x=x, y=y, metadata=metadata, feature_names=tuple(f"{FEATURE_PREFIX}{idx}" for idx in range(feature_count)))
        validate_model_input_contract(batch)
        return batch

