from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def test_run_real_classifier_route_records_filterbank_provenance(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import hust_bci_er.training.real_adapter as real_adapter

    route = tmp_path / "sliding_window_fbstcnet.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: sliding_window_fbstcnet",
                "status: IDEA",
                "dataset_version: train_v1",
                "split_id: p1_seed42_fold0",
                "seed: 42",
                "input_window_sec: 1",
                "preprocessing: [zscore]",
                "features: []",
                "model:",
                "  name: fbstcnet",
                "  sfreq: 250",
                "  filterbank_type: cheby2",
                "  filter_order: 4",
                "  stopband_ripple: 30.0",
                "  transition_bandwidth: 2.0",
                "adaptation: none",
                "training:",
                "  trainer: torch_classifier",
                "  job_adapter: torch_classifier",
                "  epochs: 1",
                "  batch_size: 2",
                "  optimizer:",
                "    name: adamw",
                "    lr: 0.001",
                "  loss: cross_entropy",
                "inference:",
                "  top4: true",
                "evaluation:",
                "  protocol: p1_repeated_group_kfold",
                "  primary_metric: exact_single_crop_expected_BA",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model_state.local.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    artifacts = SimpleNamespace(run_dir=run_dir, manifest_json=run_dir / "manifest.json")

    monkeypatch.setattr(
        real_adapter._base,
        "run_real_classifier_route",
        lambda *args, **kwargs: artifacts,
    )

    real_adapter.run_real_classifier_route(route_config_path=route, run_dir=run_dir)

    model_state = json.loads((run_dir / "model_state.local.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert model_state["model_kwargs"]["filterbank_type"] == "cheby2"
    assert model_state["filterbank_params"]["implementation"] == "magnitude_response_fft"
    assert manifest["model_filterbank_params"]["phase_response"] == "discarded"
