"""Subject-level split assignment helpers for protocol jobs."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def cohort_for_subject(subject_id: str, row: Mapping[str, Any] | None = None) -> str:
    if row is not None and row.get("cohort"):
        return str(row["cohort"])
    if subject_id.startswith("DEP"):
        return "DEP"
    if subject_id.startswith("HC"):
        return "HC"
    return "unknown"


def subjects_by_cohort(trial_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    cohorts: dict[str, set[str]] = defaultdict(set)
    for row in trial_rows:
        subject = str(row.get("subject_id") or "")
        if not subject:
            raise ValueError("trial row is missing subject_id")
        cohorts[cohort_for_subject(subject, row)].add(subject)
    return {cohort: sorted(subjects) for cohort, subjects in sorted(cohorts.items())}


def fold_buckets(subjects: Sequence[str], *, seed: int, n_folds: int) -> list[list[str]]:
    if n_folds <= 1:
        raise ValueError("n_folds must be greater than 1")
    clean = sorted(map(str, subjects))
    if len(clean) < n_folds:
        raise ValueError("n_folds must not exceed the number of subjects in any cohort")
    rng = random.Random(int(seed))
    shuffled = list(clean)
    rng.shuffle(shuffled)
    buckets = [[] for _ in range(int(n_folds))]
    for idx, subject in enumerate(shuffled):
        buckets[idx % int(n_folds)].append(subject)
    return [sorted(bucket) for bucket in buckets]


def validation_subjects_from_training_pool(
    train_pool: Sequence[str],
    *,
    seed: int,
    val_fraction: float,
) -> tuple[list[str], list[str]]:
    pool = sorted(map(str, train_pool))
    if len(pool) < 2:
        return pool, []
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    val_count = max(1, int(round(len(pool) * float(val_fraction))))
    val_count = min(val_count, len(pool) - 1)
    rng = random.Random(int(seed))
    shuffled = list(pool)
    rng.shuffle(shuffled)
    val_subjects = sorted(shuffled[:val_count])
    train_subjects = sorted(shuffled[val_count:])
    return train_subjects, val_subjects


def p1_subject_split(
    trial_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    fold: int,
    n_folds: int,
    val_fraction: float = 0.2,
) -> tuple[list[str], list[str], list[str]]:
    if fold < 0 or fold >= n_folds:
        raise ValueError("fold must be in [0, n_folds)")
    train_subjects: list[str] = []
    val_subjects: list[str] = []
    test_subjects: list[str] = []
    for cohort, subjects in subjects_by_cohort(trial_rows).items():
        buckets = fold_buckets(subjects, seed=seed, n_folds=n_folds)
        cohort_test = buckets[int(fold)]
        remaining = [subject for idx, bucket in enumerate(buckets) if idx != int(fold) for subject in bucket]
        cohort_train, cohort_val = validation_subjects_from_training_pool(
            remaining,
            seed=int(seed) + int(fold) * 1009 + sum(ord(ch) for ch in cohort),
            val_fraction=val_fraction,
        )
        train_subjects.extend(cohort_train)
        val_subjects.extend(cohort_val)
        test_subjects.extend(cohort_test)
    train_subjects = sorted(train_subjects)
    val_subjects = sorted(val_subjects)
    test_subjects = sorted(test_subjects)
    if not train_subjects or not val_subjects or not test_subjects:
        raise ValueError("P1 split requires non-empty train, val, and test subjects")
    return train_subjects, val_subjects, test_subjects


def assign_trial_rows_to_split(
    trial_rows: Sequence[Mapping[str, Any]],
    *,
    train_subjects: Sequence[str],
    val_subjects: Sequence[str],
    test_subjects: Sequence[str],
) -> list[dict[str, Any]]:
    train_set = set(map(str, train_subjects))
    val_set = set(map(str, val_subjects))
    test_set = set(map(str, test_subjects))
    out: list[dict[str, Any]] = []
    for row in trial_rows:
        subject = str(row.get("subject_id") or "")
        if subject in train_set:
            split = "train"
        elif subject in val_set:
            split = "val"
        elif subject in test_set:
            split = "test"
        else:
            raise ValueError(f"subject is not assigned to any split: {subject}")
        trial_id = row.get("trial_id")
        original_trial_id = row.get("original_trial_id") or (f"{subject}::{trial_id}" if trial_id is not None else None)
        if not original_trial_id:
            raise ValueError(f"trial row for subject {subject} has no trial_id/original_trial_id")
        item = {
            "subject_id": subject,
            "original_trial_id": str(original_trial_id),
            "split": split,
        }
        if trial_id is not None:
            item["trial_id"] = str(trial_id)
        if "cohort" in row:
            item["cohort"] = str(row["cohort"])
        out.append(item)
    return out
