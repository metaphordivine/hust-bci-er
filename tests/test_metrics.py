import numpy as np

from hust_bci_er.evaluation.metrics import all_correct_rate, balanced_accuracy


def test_balanced_accuracy():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    assert balanced_accuracy(y_true, y_pred) == 0.75


def test_all_correct_rate():
    y_true = np.array([[0, 1], [1, 0]])
    y_pred = np.array([[0, 1], [1, 1]])
    assert all_correct_rate(y_true, y_pred) == 0.5

