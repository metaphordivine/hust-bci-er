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


def _allocate_holdout_counts_by_cohort(
    cohorts: Mapping[str, Sequence[str]],
    *,
    n_holdout_subjects: int,
) -> dict[str, int]:
    total_subjects = sum(len(subjects) for subjects in cohorts.values())
    if n_holdout_subjects >= total_subjects:
        raise ValueError("n_holdout_subjects must be less than the total number of subjects")
    if not cohorts:
        raise ValueError("trial_rows must contain subjects")

    minimum = 1 if n_holdout_subjects >= len(cohorts) else 0
    raw_counts = {
        cohort: (len(subjects) * float(n_holdout_subjects)) / float(total_subjects)
        for cohort, subjects in cohorts.items()
    }
    counts = {
        cohort: min(len(cohorts[cohort]), max(minimum, int(raw_counts[cohort])))
        for cohort in cohorts
    }

    def add_order() -> list[str]:
        return sorted(
            cohorts,
            key=lambda cohort: (
                raw_counts[cohort] - int(raw_counts[cohort]),
                len(cohorts[cohort]) - counts[cohort],
                cohort,
            ),
            reverse=True,
        )

    while sum(counts.values()) < n_holdout_subjects:
        for cohort in add_order():
            if counts[cohort] < len(cohorts[cohort]):
                counts[cohort] += 1
                break
        else:
            raise ValueError("could not allocate requested P2 holdout subjects")

    while sum(counts.values()) > n_holdout_subjects:
        removable = [
            cohort
            for cohort, count in counts.items()
            if count > (1 if n_holdout_subjects >= len(cohorts) else 0)
        ]
        if not removable:
            raise ValueError("could not allocate requested P2 holdout subjects")
        cohort = min(
            removable,
            key=lambda item: (
                raw_counts[item] - int(raw_counts[item]),
                counts[item],
                item,
            ),
        )
        counts[cohort] -= 1
    return counts


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


def p2_subject_split(
    trial_rows: Sequence[Mapping[str, Any]],
    *,
    holdout_seed: int,
    train_seed: int,
    n_holdout_subjects: int,
    val_fraction: float = 0.2,
) -> tuple[list[str], list[str], list[str]]:
    """Generate a single P2 holdout split.

    Returns (train_subjects, val_subjects, test_subjects) where test_subjects
    are the holdout set and train/val are split from the remaining subjects.
    """
    if n_holdout_subjects < 1:
        raise ValueError("n_holdout_subjects must be >= 1")
    cohorts = subjects_by_cohort(trial_rows)
    all_subjects = sorted({subject for subjects in cohorts.values() for subject in subjects})
    if n_holdout_subjects >= len(all_subjects):
        raise ValueError("n_holdout_subjects must be less than the total number of subjects")

    holdout_counts = _allocate_holdout_counts_by_cohort(
        cohorts,
        n_holdout_subjects=int(n_holdout_subjects),
    )
    train_subjects: list[str] = []
    val_subjects: list[str] = []
    test_subjects: list[str] = []
    for cohort, subjects in cohorts.items():
        rng_holdout = random.Random(int(holdout_seed) + sum(ord(ch) for ch in cohort))
        shuffled = list(subjects)
        rng_holdout.shuffle(shuffled)
        cohort_test = sorted(shuffled[: holdout_counts[cohort]])
        cohort_remaining = sorted(shuffled[holdout_counts[cohort] :])
        cohort_train, cohort_val = validation_subjects_from_training_pool(
            cohort_remaining,
            seed=int(train_seed) + sum(ord(ch) for ch in cohort),
            val_fraction=val_fraction,
        )
        train_subjects.extend(cohort_train)
        val_subjects.extend(cohort_val)
        test_subjects.extend(cohort_test)
    train_subjects = sorted(train_subjects)
    val_subjects = sorted(val_subjects)
    test_subjects = sorted(test_subjects)
    if not train_subjects or not val_subjects or not test_subjects:
        raise ValueError("P2 split requires non-empty train, val, and test subjects")
    return train_subjects, val_subjects, test_subjects


def p3_outer_subject_split(
    trial_rows: Sequence[Mapping[str, Any]],
    *,
    outer_seed: int,
    outer_fold: int,
    outer_folds: int,
    val_fraction: float = 0.2,
) -> tuple[list[str], list[str], list[str]]:
    """Generate the P3 final outer train/val/test subject split."""
    if outer_fold < 0 or outer_fold >= outer_folds:
        raise ValueError("outer_fold must be in [0, outer_folds)")

    train_subjects: list[str] = []
    val_subjects: list[str] = []
    test_subjects: list[str] = []
    for cohort, subjects in subjects_by_cohort(trial_rows).items():
        outer_buckets = fold_buckets(subjects, seed=int(outer_seed), n_folds=int(outer_folds))
        cohort_test = outer_buckets[int(outer_fold)]
        outer_remaining = [s for idx, bucket in enumerate(outer_buckets) if idx != int(outer_fold) for s in bucket]
        cohort_train, cohort_val = validation_subjects_from_training_pool(
            outer_remaining,
            seed=int(outer_seed) + int(outer_fold) * 1009 + sum(ord(ch) for ch in cohort),
            val_fraction=val_fraction,
        )
        train_subjects.extend(cohort_train)
        val_subjects.extend(cohort_val)
        test_subjects.extend(cohort_test)

    train_subjects = sorted(train_subjects)
    val_subjects = sorted(val_subjects)
    test_subjects = sorted(test_subjects)
    if not train_subjects or not val_subjects or not test_subjects:
        raise ValueError("P3 outer split requires non-empty train, val, and test subjects")
    return train_subjects, val_subjects, test_subjects


def p3_outer_inner_split(
    trial_rows: Sequence[Mapping[str, Any]],
    *,
    outer_seed: int,
    inner_seed: int,
    outer_fold: int,
    inner_fold: int,
    outer_folds: int,
    inner_folds: int,
) -> tuple[list[str], list[str], list[str]]:
    """Generate a nested P3 outer/inner subject split.

    Outer fold defines a holdout test set (like P1 fold).
    Inner fold splits the remaining subjects for hyperparameter selection:
    the inner-fold test subjects serve as the inner validation set.
    """
    if outer_fold < 0 or outer_fold >= outer_folds:
        raise ValueError("outer_fold must be in [0, outer_folds)")
    if inner_fold < 0 or inner_fold >= inner_folds:
        raise ValueError("inner_fold must be in [0, inner_folds)")
    if inner_folds < 2:
        raise ValueError("inner_folds must be >= 2")

    train_subjects: list[str] = []
    val_subjects: list[str] = []
    test_subjects: list[str] = []
    for cohort, subjects in subjects_by_cohort(trial_rows).items():
        outer_buckets = fold_buckets(subjects, seed=int(outer_seed), n_folds=int(outer_folds))
        cohort_test = outer_buckets[int(outer_fold)]
        outer_remaining = [s for idx, bucket in enumerate(outer_buckets) if idx != int(outer_fold) for s in bucket]

        inner_buckets = fold_buckets(outer_remaining, seed=int(inner_seed), n_folds=int(inner_folds))
        cohort_val = inner_buckets[int(inner_fold)]
        cohort_train = [s for idx, bucket in enumerate(inner_buckets) if idx != int(inner_fold) for s in bucket]

        train_subjects.extend(cohort_train)
        val_subjects.extend(cohort_val)
        test_subjects.extend(cohort_test)

    train_subjects = sorted(train_subjects)
    val_subjects = sorted(val_subjects)
    test_subjects = sorted(test_subjects)
    if not train_subjects or not val_subjects or not test_subjects:
        raise ValueError("P3 inner split requires non-empty train, val, and test subjects")
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
