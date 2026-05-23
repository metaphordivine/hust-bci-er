from __future__ import annotations

import zlib
import json

import numpy as np
import pytest

from hust_bci_er.analysis.dep_hc_router import RouterSample
from hust_bci_er.tasks.dep_hc.experiment import evaluate_dep_hc_task, extract_dep_hc_task_features, write_dep_hc_task_outputs
from hust_bci_er.tasks.dep_hc.channel_graph import (
    channel_graph_adjacency,
    channel_graph_metadata,
    graph_edge_indices,
    regional_connectivity_features,
)
from hust_bci_er.tasks.dep_hc.features import (
    asymmetry_features,
    connectivity_summary_features,
    dep_hc_features,
    time_frequency_summary_features,
)


def _window(seed_text: str, *, scale: float = 1.0) -> np.ndarray:
    seed = zlib.crc32(seed_text.encode("utf-8"))
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=(30, 256)).astype(np.float32)


def _sample(subject: str, cohort: str, *, crop_id: int = 0) -> RouterSample:
    scale = 0.5 if cohort == "HC" else 1.4
    return RouterSample(
        x=_window(f"{subject}-{cohort}-{crop_id}", scale=scale),
        subject_id=subject,
        trial_id=f"{subject}_t{crop_id}",
        crop_id=crop_id,
        window_start_sec=float(crop_id),
        cohort=cohort,
    )


def test_dep_hc_task_feature_sets_are_numeric_and_metadata_free():
    x = _window("feature-smoke")

    traditional = dep_hc_features(x, feature_set="traditional", sfreq=128.0)
    traditional_graph = dep_hc_features(x, feature_set="traditional_graph", sfreq=128.0)
    connectivity = connectivity_summary_features(x)
    regional = regional_connectivity_features(x)
    asymmetry = asymmetry_features(x, sfreq=128.0)
    time_frequency = time_frequency_summary_features(x, sfreq=128.0)

    assert traditional.ndim == 1
    assert connectivity.shape == (30 * 29,)
    assert regional.shape == (30,)
    assert asymmetry.shape == (13,)
    assert time_frequency.shape == (15,)
    assert traditional_graph.shape[0] == traditional.shape[0] + regional.shape[0]
    assert np.isfinite(traditional).all()


def test_traditional_asymmetry_requires_hust_montage():
    with pytest.raises(ValueError, match="hust_30_a2"):
        dep_hc_features(_window("bad-montage"), feature_set="traditional", channel_montage="sequential")


def test_hust_channel_graph_uses_pdf_order_and_region_priors():
    table = channel_graph_metadata()["channels"]
    adjacency = channel_graph_adjacency()
    edges = graph_edge_indices(include_self=False)

    assert table[0]["name"] == "FP1"
    assert table[-1]["name"] == "O2"
    assert adjacency.shape == (30, 30)
    assert np.allclose(adjacency, adjacency.T)
    assert len(edges) > 30
    assert adjacency[0, 1] == 1.0
    assert adjacency[3, 5] == 1.0


def test_dep_hc_task_feature_matrix_and_eval():
    train = [
        _sample("HC001", "HC"),
        _sample("HC002", "HC"),
        _sample("DEP001", "DEP"),
        _sample("DEP002", "DEP"),
    ]
    eval_samples = [
        _sample("HC101", "HC", crop_id=0),
        _sample("HC101", "HC", crop_id=1),
        _sample("DEP101", "DEP", crop_id=0),
        _sample("DEP101", "DEP", crop_id=1),
    ]

    features = extract_dep_hc_task_features(train, feature_set="traditional_graph", sfreq=128.0)
    result = evaluate_dep_hc_task(
        train,
        eval_samples,
        feature_set="traditional_graph",
        sfreq=128.0,
        epochs=40,
        lr=0.1,
        l2=1e-3,
    )

    assert features.shape[0] == 4
    assert result["metrics"]["task"] == "dep_hc"
    assert result["metrics"]["feature_set"] == "traditional_graph"
    assert result["metrics"]["n_eval_subjects"] == 2
    assert {"DEP101", "HC101"} == {row["subject_id"] for row in result["subject_rows"]}


def test_write_dep_hc_task_outputs_uses_task_specific_payload(tmp_path):
    result = {
        "metrics": {"task": "dep_hc", "subject_ba": 1.0},
        "prediction_rows": [
            {
                "subject_id": "DEP001",
                "trial_id": "DEP001_t0",
                "crop_id": "0",
                "window_start_sec": "0.0",
                "cohort": "DEP",
                "y_true": "1",
                "p_hc": "0.1",
                "p_dep": "0.9",
                "confidence": "0.9",
                "y_pred": "1",
                "predicted_cohort": "DEP",
            }
        ],
        "subject_rows": [
            {
                "subject_id": "DEP001",
                "cohort": "DEP",
                "y_true": "1",
                "mean_p_dep": "0.9",
                "mean_p_hc": "0.1",
                "threshold": "0.5",
                "confidence": "0.9",
                "predicted_cohort": "DEP",
                "correct": "1",
                "n_windows": "1",
            }
        ],
    }

    write_dep_hc_task_outputs(result, tmp_path, config={"feature_set": "traditional"})

    payload = json.loads((tmp_path / "dep_hc_task_diagnostic.json").read_text(encoding="utf-8"))
    assert payload["task"] == "dep_hc"
    assert (tmp_path / "dep_hc_predictions.csv").exists()
    assert (tmp_path / "dep_hc_subject_metrics.csv").exists()
