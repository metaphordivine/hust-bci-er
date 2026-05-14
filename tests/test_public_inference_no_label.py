from inspect import signature
from pathlib import Path

from hust_bci_er.inference.topk import subject_top4


FORBIDDEN_TERMS = ["y_true", "label", "ground_truth", "public_truth", "private_truth"]


def test_subject_top4_does_not_accept_labels():
    params = set(signature(subject_top4).parameters)
    assert "y_true" not in params
    assert "label" not in params


def test_inference_runtime_code_does_not_reference_labels():
    hits = []
    for path in Path("src/hust_bci_er/inference").glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for term in FORBIDDEN_TERMS:
            if term in text:
                hits.append(f"{path}: {term}")
    assert hits == []
