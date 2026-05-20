from pathlib import Path

import numpy as np
import pytest
import yaml

from hust_bci_er.training.real_adapter import run_real_classifier_route


torch = pytest.importorskip("torch")


def _write_hdf5_mat(path: Path, *, samples_per_trial: int = 50) -> None:
    h5py = pytest.importorskip("h5py")
    total = samples_per_trial * 4
    neu = np.zeros((30, total), dtype=np.float32)
    pos = np.ones((30, total), dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("EEG_data_neu", data=neu)
        f.create_dataset("EEG_data_pos", data=pos)


def _write_fake_data(data_root: Path) -> None:
    data_root.mkdir()
    for idx in range(3):
        _write_hdf5_mat(data_root / f"DEP{idx:03d}timedata.mat")
        _write_hdf5_mat(data_root / f"HC{idx:03d}timedata.mat")


def _model_config(model_name: str) -> dict:
    configs = {
        "tsception": {
            "name": "tsception",
            "channel_montage": "hust_30_a2",
            "n_filters": 2,
            "temporal_kernel_sizes": [3, 5],
            "temporal_pool_size": 1,
            "classifier_hidden_dim": 8,
            "dropout": 0.1,
        },
        "fbcnet": {
            "name": "fbcnet",
            "n_bands": 3,
            "spatial_filters": 1,
            "temporal_kernel_size": 3,
            "n_segments": 2,
            "classifier_hidden_dim": 8,
            "dropout": 0.1,
        },
        "dgcnn": {
            "name": "dgcnn",
            "channel_montage": "hust_30_a2",
            "node_features": 4,
            "graph_hidden_dim": 4,
            "k_order": 2,
            "temporal_kernel_size": 3,
            "classifier_hidden_dim": 8,
            "dropout": 0.1,
        },
        "lggnet": {
            "name": "lggnet",
            "channel_montage": "hust_30_a2",
            "n_regions": 5,
            "temporal_filters": 2,
            "temporal_kernel_sizes": [3, 5],
            "temporal_pool_size": 1,
            "graph_hidden_dim": 4,
            "classifier_hidden_dim": 8,
            "dropout": 0.1,
        },
        "riemannian_tangent": {
            "name": "riemannian_tangent",
            "covariance_eps": 0.001,
            "shrinkage": 0.1,
            "feature_standardize": True,
            "dropout": 0.0,
        },
    }
    return configs[model_name]


def _write_route(path: Path, model_name: str) -> None:
    route_id = path.stem
    route = {
        "route_id": route_id,
        "status": "IDEA",
        "dataset_version": "train_v1",
        "split_id": "p1_seed42_fold0",
        "seed": 42,
        "input_window_sec": 0.04,
        "preprocessing": ["zscore"],
        "features": [],
        "augmentation": {
            "name": "split_first_fixed_crops",
            "split_first": True,
            "source_trial_sec": 0.2,
            "window_sec": 0.04,
            "n_crops": 5,
            "apply_to_splits": ["train", "val", "test"],
            "aggregate_to_trial": {"method": "mean_score", "tie_break": "mean_score"},
        },
        "model": _model_config(model_name),
        "adaptation": "none",
        "training": {
            "trainer": "torch_classifier",
            "job_adapter": "torch_classifier",
            "epochs": 1,
            "batch_size": 4,
            "optimizer": {"name": "sgd", "lr": 0.01, "weight_decay": 0.0},
            "loss": "cross_entropy",
        },
        "inference": {"top4": True, "crop_policy": "exact_single_crop"},
        "evaluation": {
            "protocol": "p1_repeated_group_kfold",
            "primary_metric": "exact_single_crop_expected_BA",
        },
    }
    path.write_text(yaml.safe_dump(route, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize("model_name", ["tsception", "fbcnet", "dgcnn", "lggnet", "riemannian_tangent"])
def test_new_routes_run_one_epoch_diagnostic_smoke(tmp_path, monkeypatch, model_name):
    import hust_bci_er.training.reproducibility as reproducibility

    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setattr(reproducibility, "PROCESS_START_PYTHONHASHSEED", "42")

    data_root = tmp_path / "data"
    _write_fake_data(data_root)
    route = tmp_path / f"diagnostic_{model_name}.yaml"
    _write_route(route, model_name)

    artifacts = run_real_classifier_route(
        route_config_path=route,
        run_dir=tmp_path / "run",
        run_mode="smoke",
        command=["pytest", "diagnostic"],
        data_root=data_root,
        device="cpu",
    )

    assert artifacts.prediction_csv.exists()
    assert artifacts.score_matrix_csv.exists()
    manifest = yaml.safe_load(artifacts.manifest_json.read_text(encoding="utf-8"))
    assert manifest["run_mode"] == "smoke"
    assert manifest["prediction_scope"] == "val_only"
    assert manifest["model_kwargs"] == {key: value for key, value in _model_config(model_name).items() if key != "name"}
