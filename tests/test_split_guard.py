import pytest

from hust_bci_er.data.splits import assert_disjoint_subjects, assert_original_trial_not_cross_split


def test_disjoint_subjects_guard():
    assert_disjoint_subjects({"s1"}, {"s2"}, {"s3"})
    with pytest.raises(ValueError):
        assert_disjoint_subjects({"s1"}, {"s1"}, {"s3"})


def test_original_trial_not_cross_split_guard():
    rows = [
        {"subject_id": "s1", "original_trial_id": "t1", "split": "train"},
        {"subject_id": "s1", "original_trial_id": "t1", "split": "train"},
    ]
    assert_original_trial_not_cross_split(rows)
    rows.append({"subject_id": "s1", "original_trial_id": "t1", "split": "test"})
    with pytest.raises(ValueError):
        assert_original_trial_not_cross_split(rows)

