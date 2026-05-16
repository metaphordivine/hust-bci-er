"""Lightweight HUST EEG .mat trial index helpers.

These helpers inspect MATLAB v7.3/HDF5 files enough to build auditable
subject/trial split contracts without loading full EEG arrays into memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


KEY_TO_LABEL = {"EEG_data_neu": 0, "EEG_data_pos": 1}
SOURCE_TRIALS_PER_CLASS = 4
N_CHANNELS = 30


def subject_and_cohort_from_mat_path(path: Path) -> tuple[str, str]:
    subject_id = path.stem.removesuffix("timedata")
    if subject_id.startswith("DEP"):
        return subject_id, "DEP"
    if subject_id.startswith("HC"):
        return subject_id, "HC"
    raise ValueError(f"unknown HUST EEG subject prefix in {path.name}; expected DEP* or HC*")


def mat_dataset_shape(path: Path, key: str) -> tuple[int, int]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to index HUST EEG .mat files.") from exc
    try:
        with h5py.File(path, "r") as f:
            if key not in f:
                raise KeyError(f"{path.name} is missing dataset {key}")
            shape = tuple(int(dim) for dim in f[key].shape)
    except OSError as exc:
        raise ValueError(f"{path} is not an HDF5/v7.3 MATLAB .mat file readable by h5py") from exc
    if len(shape) != 2:
        raise ValueError(f"unexpected data rank in {path.name}:{key}: {shape}")
    if shape[0] == N_CHANNELS:
        return shape[0], shape[1]
    if shape[1] == N_CHANNELS:
        return shape[1], shape[0]
    raise ValueError(f"unexpected data shape in {path.name}:{key}: {shape}")


def hust_mat_trial_index(data_root: Path) -> list[dict[str, Any]]:
    """Return one row per source trial under ``data_root``.

    The row shape intentionally mirrors the run-local split evidence generated
    by the real adapter: ``original_trial_id`` is ``subject_id::trial_id``.
    """

    data_root = Path(data_root).resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(data_root.rglob("*timedata.mat")):
        subject_id, cohort = subject_and_cohort_from_mat_path(path)
        for key, label in KEY_TO_LABEL.items():
            _n_channels, n_samples = mat_dataset_shape(path, key)
            if n_samples % SOURCE_TRIALS_PER_CLASS != 0:
                raise ValueError(f"{path.name}:{key} is not divisible into {SOURCE_TRIALS_PER_CLASS} trials")
            samples_per_trial = n_samples // SOURCE_TRIALS_PER_CLASS
            label_name = "pos" if label == 1 else "neu"
            for idx in range(SOURCE_TRIALS_PER_CLASS):
                trial_id = f"{subject_id}_{label_name}{idx + 1}"
                rows.append(
                    {
                        "subject_id": subject_id,
                        "cohort": cohort,
                        "trial_id": trial_id,
                        "original_trial_id": f"{subject_id}::{trial_id}",
                        "raw_path": str(path),
                        "mat_key": key,
                        "source_trial_index": idx,
                        "sampling_rate_hz": 250,
                        "n_channels": N_CHANNELS,
                        "n_samples": samples_per_trial,
                        "y_true": label,
                    }
                )
    if not rows:
        raise FileNotFoundError(f"no *timedata.mat files found under {data_root}")
    return rows
