import numpy as np

from hust_bci_er.evaluation.exact_single_crop import (
    assignment_grid,
    exact_all_correct_rate_from_matrix,
    exact_ba_from_matrix,
)


def test_assignment_grid_shape():
    assert assignment_grid().shape == (5**8, 8)


def test_exact_ba_perfect_constant_ranking():
    mat = np.tile(np.array([[8], [7], [6], [5], [4], [3], [2], [1]], dtype=float), (1, 5))
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert exact_ba_from_matrix(mat, y_true) == 1.0
    assert exact_all_correct_rate_from_matrix(mat, y_true) == 1.0

