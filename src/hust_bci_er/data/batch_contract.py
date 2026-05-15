"""Runtime checks that keep model inputs separate from audit metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from hust_bci_er.contracts.prediction import MODEL_FORBIDDEN_COLUMNS


@dataclass(frozen=True)
class ModelBatch:
    x: np.ndarray
    y: np.ndarray | None
    metadata: Mapping[str, Sequence[Any]]
    feature_names: tuple[str, ...] = ()


def validate_model_input_contract(batch: ModelBatch) -> None:
    """Reject batches that expose forbidden metadata as model features."""
    if not isinstance(batch.x, np.ndarray):
        raise TypeError("batch.x must be a numpy array")
    if batch.x.ndim < 2:
        raise ValueError("batch.x must include a batch dimension and a feature dimension")
    if batch.y is not None and len(batch.y) != len(batch.x):
        raise ValueError("batch.y length must match batch.x")

    metadata_keys = {str(key) for key in batch.metadata}
    if not metadata_keys:
        raise ValueError("batch.metadata must declare grouping/audit fields")
    missing = {"subject_id", "trial_id"} - metadata_keys
    if missing:
        raise ValueError("batch.metadata missing required audit keys: " + ", ".join(sorted(missing)))

    feature_names = {str(name) for name in batch.feature_names}
    forbidden_features = sorted(feature_names & MODEL_FORBIDDEN_COLUMNS)
    if forbidden_features:
        raise ValueError("forbidden metadata columns used as model features: " + ", ".join(forbidden_features))

