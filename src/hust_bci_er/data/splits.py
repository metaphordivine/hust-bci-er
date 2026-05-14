"""Subject/trial split helpers.

Important guardrail: create subject/trial splits before any overlapping window
augmentation. Windowing before split can leak overlapping signal into test data.
This module does not implement sliding-window augmentation; it only provides
split integrity checks.
"""

from __future__ import annotations


def assert_disjoint_subjects(train_subjects: set[str], val_subjects: set[str], test_subjects: set[str]) -> None:
    overlaps = {
        "train_val": train_subjects & val_subjects,
        "train_test": train_subjects & test_subjects,
        "val_test": val_subjects & test_subjects,
    }
    bad = {name: sorted(values) for name, values in overlaps.items() if values}
    if bad:
        raise ValueError(f"subject split leakage: {bad}")


def assert_original_trial_not_cross_split(rows: list[dict]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (str(row["subject_id"]), str(row["original_trial_id"]))
        split = str(row["split"])
        if key in seen and seen[key] != split:
            raise ValueError(f"original trial crosses split: {key} in {seen[key]} and {split}")
        seen[key] = split

