"""Build deterministic subject-level split manifests."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


def load_dataset_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"dataset manifest must be a mapping: {path}")
    return data


def trial_rows_from_dataset(dataset: Mapping[str, Any]) -> Sequence[Any]:
    rows = dataset.get("trial_index") or dataset.get("trial_rows") or []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return rows


def subject_ids_from_dataset(dataset: Mapping[str, Any]) -> list[str]:
    rows = trial_rows_from_dataset(dataset)
    subjects = sorted({str(row["subject_id"]) for row in rows if isinstance(row, Mapping) and row.get("subject_id")})
    if not subjects:
        raise ValueError("dataset manifest has no subject_id values")
    return subjects


def original_trial_id_from_row(row: Mapping[str, Any], *, index: int) -> str:
    original_trial_id = row.get("original_trial_id", row.get("trial_id"))
    if original_trial_id is None or str(original_trial_id) == "":
        raise ValueError(f"dataset row {index} has no original_trial_id or trial_id")
    return str(original_trial_id)


def split_subjects(subjects: Sequence[str], *, seed: int, val_count: int, test_count: int) -> tuple[list[str], list[str], list[str]]:
    subjects = sorted(map(str, subjects))
    if val_count < 0 or test_count < 0:
        raise ValueError("val_count and test_count must be non-negative")
    if val_count + test_count >= len(subjects):
        raise ValueError("val_count + test_count must leave at least one train subject")
    rng = random.Random(int(seed))
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    test_subjects = sorted(shuffled[:test_count])
    val_subjects = sorted(shuffled[test_count : test_count + val_count])
    train_subjects = sorted(shuffled[test_count + val_count :])
    return train_subjects, val_subjects, test_subjects


def split_for_subject(subject: str, *, train_subjects: set[str], val_subjects: set[str], test_subjects: set[str]) -> str:
    if subject in train_subjects:
        return "train"
    if subject in val_subjects:
        return "val"
    if subject in test_subjects:
        return "test"
    raise ValueError(f"subject is not assigned to a split: {subject}")


def build_split_manifest(
    *,
    split_id: str,
    dataset: Mapping[str, Any],
    seed: int,
    val_count: int,
    test_count: int,
    dataset_manifest_path: str | None = None,
    dataset_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    subjects = subject_ids_from_dataset(dataset)
    train_subjects, val_subjects, test_subjects = split_subjects(subjects, seed=seed, val_count=val_count, test_count=test_count)
    sets = (set(train_subjects), set(val_subjects), set(test_subjects))
    trial_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(trial_rows_from_dataset(dataset)):
        if not isinstance(row, Mapping):
            continue
        subject_value = row.get("subject_id")
        if subject_value is None or str(subject_value) == "":
            raise ValueError(f"dataset row {idx} has no subject_id")
        subject = str(subject_value)
        trial_rows.append(
            {
                "subject_id": subject,
                "original_trial_id": original_trial_id_from_row(row, index=idx),
                "split": split_for_subject(subject, train_subjects=sets[0], val_subjects=sets[1], test_subjects=sets[2]),
            }
        )
    return {
        "split_id": split_id,
        "subject_group_split": True,
        "status": "ready",
        "dataset_version": dataset.get("dataset_version"),
        "dataset_manifest_path": dataset_manifest_path,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "seed": int(seed),
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "trial_rows": trial_rows,
    }


def write_split_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(manifest), sort_keys=False, allow_unicode=True), encoding="utf-8")
