from inspect import signature

from hust_bci_er.inference.topk import subject_top4


def test_subject_top4_does_not_accept_labels():
    params = set(signature(subject_top4).parameters)
    assert "y_true" not in params
    assert "label" not in params

