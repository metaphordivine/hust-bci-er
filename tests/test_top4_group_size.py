import numpy as np
import pytest

from hust_bci_er.inference.topk import subject_top4


def test_subject_top4_marks_exactly_four_trials():
    pred = subject_top4(np.arange(8))
    assert pred.sum() == 4
    assert pred.shape == (8,)


def test_subject_top4_requires_eight_trials():
    with pytest.raises(ValueError):
        subject_top4(np.arange(7))

