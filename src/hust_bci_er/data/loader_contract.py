"""Dataset loader contracts for raw trials and crops."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from hust_bci_er.contracts.records import EEGTrial, EEGWindow


@dataclass(frozen=True)
class TrialCrop:
    x: np.ndarray
    subject_id: str
    trial_id: str
    crop_id: int
    y: int | None = None
    split: str | None = None
    path: str | None = None
    metadata: Mapping[str, Any] | None = None


class DatasetLoader(Protocol):
    def iter_trials(self) -> Iterator[EEGTrial]:
        ...

    def iter_crops(self) -> Iterator[TrialCrop]:
        ...

    def iter_windows(self) -> Iterator[EEGWindow]:
        ...


def trial_crop_from_index_row(row: Mapping[str, Any], *, x: np.ndarray | None = None) -> TrialCrop:
    return TrialCrop(
        x=np.asarray(x if x is not None else np.empty((0,), dtype=np.float32)),
        subject_id=str(row["subject_id"]),
        trial_id=str(row["trial_id"]),
        crop_id=int(row.get("crop_id", 0)),
        y=int(row["y_true"]) if row.get("y_true") not in {None, ""} else None,
        split=str(row["split"]) if row.get("split") not in {None, ""} else None,
        path=str(row["path"]) if row.get("path") not in {None, ""} else None,
        metadata=dict(row),
    )


def crops_from_manifest(manifest: Mapping[str, Any]) -> list[TrialCrop]:
    rows = manifest.get("trial_index") or []
    if not isinstance(rows, Sequence):
        raise ValueError("manifest trial_index must be a sequence")
    return [trial_crop_from_index_row(row) for row in rows if isinstance(row, Mapping)]


def resolve_crop_path(crop: TrialCrop, *, base_dir: Path) -> Path | None:
    if crop.path is None:
        return None
    path = Path(crop.path)
    return path if path.is_absolute() else base_dir / path
