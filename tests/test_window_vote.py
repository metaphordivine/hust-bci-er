import pytest

from hust_bci_er.inference.window_vote import aggregate_window_predictions, strict_majority_class


def test_strict_majority_class_uses_three_of_four_windows():
    assert strict_majority_class([1, 1, 1, 0], scores=[0.9, 0.8, 0.7, 0.2]) == 1
    assert strict_majority_class([0, 0, 0, 1], scores=[0.1, 0.2, 0.3, 0.8]) == 0


def test_strict_majority_class_tie_uses_mean_score():
    assert strict_majority_class([1, 1, 0, 0], scores=[0.9, 0.7, 0.4, 0.3]) == 1
    assert strict_majority_class([1, 1, 0, 0], scores=[0.6, 0.5, 0.4, 0.3], threshold=0.5) == 0


def test_aggregate_window_predictions_returns_one_trial_row():
    rows = [
        {"subject_id": "s1", "trial_id": "t1", "crop_id": 0, "y_score": 0.9, "y_pred": 1},
        {"subject_id": "s1", "trial_id": "t1", "crop_id": 1, "y_score": 0.8, "y_pred": 1},
        {"subject_id": "s1", "trial_id": "t1", "crop_id": 2, "y_score": 0.7, "y_pred": 1},
        {"subject_id": "s1", "trial_id": "t1", "crop_id": 3, "y_score": 0.2, "y_pred": 0},
    ]

    assert aggregate_window_predictions(rows) == [
        {"subject_id": "s1", "trial_id": "t1", "y_score": pytest.approx(0.65), "y_pred": 1, "n_windows": 4}
    ]
