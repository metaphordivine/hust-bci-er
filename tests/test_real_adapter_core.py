"""Unit tests for real_adapter core helper functions.

Tests cover:
- _split_subjects: disjointness, capping, empty-train guard
- _build_score_matrix: 5-crop filling for single-crop and sliding-window policies
- _apply_preprocessing: unknown step rejection
- _fit_ea_on_windows: skips when euclidean_alignment not in preproc
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import hust_bci_er.training.real_adapter as real_adapter
from hust_bci_er.training.real_adapter import (
    FIXED_CANDIDATE_CROPS,
    KNOWN_PREPROC,
    _apply_preprocessing,
    _build_score_matrix,
    _fit_ea_on_windows,
    _load_mat_trials,
    _make_fixed_crops,
    _read_mat,
    _split_subjects,
    _validate_fixed_crop_coverage,
    _write_evidence_manifests,
    run_real_classifier_route,
)


# ---------------------------------------------------------------------------
# _split_subjects
# ---------------------------------------------------------------------------

def _make_trials(dep_n: int, hc_n: int) -> list[dict]:
    """Create synthetic trial dicts with the fields _split_subjects needs."""
    trials = []
    for i in range(dep_n):
        trials.append({"subject_id": f"DEP{i:02d}", "cohort": "DEP", "y": 0,
                        "x": np.zeros((30, 100), dtype=np.float32)})
    for i in range(hc_n):
        trials.append({"subject_id": f"HC{i:02d}", "cohort": "HC", "y": 1,
                        "x": np.zeros((30, 100), dtype=np.float32)})
    return trials


def test_split_subjects_are_mutually_disjoint():
    trials = _make_trials(dep_n=6, hc_n=10)
    train, val, test = _split_subjects(trials, val_dep=2, val_hc=3, seed=42)
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()


def test_split_subjects_covers_all():
    trials = _make_trials(dep_n=6, hc_n=10)
    train, val, test = _split_subjects(trials, val_dep=2, val_hc=3, seed=42)
    all_ids = {t["subject_id"] for t in trials}
    assert train | val | test == all_ids


def test_split_subjects_train_not_empty():
    """With enough subjects, train must be non-empty."""
    trials = _make_trials(dep_n=4, hc_n=8)
    train, val, test = _split_subjects(trials, val_dep=1, val_hc=2, seed=0)
    assert len(train) > 0


def test_split_subjects_caps_holdout_when_few_subjects():
    """With only 2 subjects per cohort, val and test get 0 each to leave train non-empty."""
    trials = _make_trials(dep_n=2, hc_n=2)
    train, val, test = _split_subjects(trials, val_dep=2, val_hc=2, seed=0)
    # n_total=2, (2-1)//2 = 0 → no holdout, all go to train
    assert len(train) == 4
    assert len(val) == 0
    assert len(test) == 0


def test_split_subjects_empty_train_raises():
    """If there are no subjects at all, train will be empty and must raise."""
    trials: list[dict] = []
    with pytest.raises(ValueError, match="training set is empty"):
        _split_subjects(trials, val_dep=0, val_hc=0, seed=0)


def test_split_subjects_is_reproducible():
    trials = _make_trials(dep_n=8, hc_n=12)
    r1 = _split_subjects(trials, val_dep=2, val_hc=3, seed=7)
    r2 = _split_subjects(trials, val_dep=2, val_hc=3, seed=7)
    assert r1 == r2


def test_split_subjects_different_seeds_differ():
    trials = _make_trials(dep_n=8, hc_n=12)
    r1 = _split_subjects(trials, val_dep=2, val_hc=3, seed=1)
    r2 = _split_subjects(trials, val_dep=2, val_hc=3, seed=2)
    # Very likely to differ with enough subjects
    assert r1[0] != r2[0] or r1[1] != r2[1] or r1[2] != r2[2]


# ---------------------------------------------------------------------------
# _build_score_matrix
# ---------------------------------------------------------------------------

def _make_window_rows(trial_id: str, n_windows: int, subject_id: str = "S01",
                      y_true: int = 1, base_score: float = 0.7) -> list[dict]:
    rows = []
    for i in range(n_windows):
        rows.append({
            "subject_id": subject_id,
            "trial_id": trial_id,
            "crop_id": i,
            "window_start_sec": float(i),
            "y_true": y_true,
            "y_score": base_score + i * 0.01,
            "y_pred": 1,
        })
    return rows


def test_build_score_matrix_single_crop_produces_5_crops():
    rows = _make_window_rows("t1", n_windows=1, base_score=0.6)
    result, evidence = _build_score_matrix(rows, crop_policy="single", seed=42)
    assert evidence == "synthetic"
    assert len(result) == 1
    assert all(f"crop_{i}" in result[0] for i in range(5))


def test_build_score_matrix_single_crop_crops_near_base():
    rows = _make_window_rows("t1", n_windows=1, base_score=0.6)
    result, _ = _build_score_matrix(rows, crop_policy="single", seed=42)
    for i in range(5):
        assert abs(float(result[0][f"crop_{i}"]) - 0.6) < 1e-4, (
            f"crop_{i} far from base score: {result[0][f'crop_{i}']}"
        )


def test_build_score_matrix_sliding_window_uses_real_scores():
    rows = _make_window_rows("t1", n_windows=5, base_score=0.5)
    result, evidence = _build_score_matrix(rows, crop_policy="sliding_window_vote", seed=42)
    assert evidence == "genuine"
    assert len(result) == 1
    for i in range(5):
        expected = f"{0.5 + i * 0.01:.8f}"
        assert result[0][f"crop_{i}"] == expected
        assert result[0][f"crop_{i}_source_crop_id"] == str(i)
        assert result[0][f"crop_{i}_window_start_sec"] == f"{float(i):.8f}"


def test_build_score_matrix_five_fixed_crops_are_genuine():
    rows = _make_window_rows("t1", n_windows=5, base_score=0.55)
    result, evidence = _build_score_matrix(rows, crop_policy="single", seed=42)
    assert evidence == "genuine"
    assert len(result) == 1
    for i in range(5):
        assert result[0][f"crop_{i}"] == f"{0.55 + i * 0.01:.8f}"


def test_build_score_matrix_preserves_y_true():
    rows = _make_window_rows("t1", n_windows=1, y_true=0)
    result, _ = _build_score_matrix(rows, crop_policy="single", seed=42)
    assert result[0]["y_true"] == 0


def test_build_score_matrix_multiple_trials():
    rows = (
        _make_window_rows("t1", n_windows=1, subject_id="S01", base_score=0.8)
        + _make_window_rows("t2", n_windows=1, subject_id="S01", base_score=0.3)
    )
    result, _ = _build_score_matrix(rows, crop_policy="single", seed=42)
    assert len(result) == 2
    trial_ids = {r["trial_id"] for r in result}
    assert trial_ids == {"t1", "t2"}


def test_build_score_matrix_sliding_window_few_windows_fallback_synthetic():
    """When sliding_window_vote has <5 windows, evidence must be synthetic."""
    rows = _make_window_rows("t1", n_windows=3, base_score=0.5)
    result, evidence = _build_score_matrix(rows, crop_policy="sliding_window_vote", seed=42)
    assert evidence == "synthetic"
    assert len(result) == 1
    assert all(f"crop_{i}" in result[0] for i in range(5))


def test_build_score_matrix_duplicate_crop_ids_fallback_synthetic():
    rows = _make_window_rows("t1", n_windows=5, base_score=0.5)
    for row in rows:
        row["crop_id"] = 0
    result, evidence = _build_score_matrix(rows, crop_policy="sliding_window_vote", seed=42)
    assert evidence == "synthetic"
    assert len(result) == 1


def test_build_score_matrix_seed_deterministic():
    rows = _make_window_rows("t1", n_windows=1, base_score=0.5)
    r1, _ = _build_score_matrix(rows, crop_policy="single", seed=99)
    r2, _ = _build_score_matrix(rows, crop_policy="single", seed=99)
    for i in range(5):
        assert r1[0][f"crop_{i}"] == r2[0][f"crop_{i}"]


def test_build_score_matrix_seed_independent_of_trial_id():
    """Crop scores must NOT depend on trial_id — only on seed and base score."""
    rows_a = _make_window_rows("trial_AAA", n_windows=1, base_score=0.5)
    rows_b = _make_window_rows("trial_ZZZ", n_windows=1, base_score=0.5)
    r_a, _ = _build_score_matrix(rows_a, crop_policy="single", seed=42)
    r_b, _ = _build_score_matrix(rows_b, crop_policy="single", seed=42)
    for i in range(5):
        assert r_a[0][f"crop_{i}"] == r_b[0][f"crop_{i}"], (
            f"crop_{i} differs by trial_id: {r_a[0][f'crop_{i}']} vs {r_b[0][f'crop_{i}']}"
        )


# ---------------------------------------------------------------------------
# _apply_preprocessing
# ---------------------------------------------------------------------------

def _random_window(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((30, 1500)).astype(np.float32)


def test_apply_preprocessing_unknown_step_raises():
    x = _random_window()
    with pytest.raises(ValueError, match="unknown preprocessing step"):
        _apply_preprocessing(x, ["whitening_eps1e-3"])  # hyphen instead of underscore


def test_apply_preprocessing_empty_list_noop():
    x = _random_window()
    out = _apply_preprocessing(x.copy(), [])
    np.testing.assert_array_equal(out, x)


def test_apply_preprocessing_zscore_normalizes_channels():
    x = _random_window()
    out = _apply_preprocessing(x.copy(), ["zscore"])
    means = out.mean(axis=1)
    stds = out.std(axis=1)
    np.testing.assert_allclose(means, 0.0, atol=1e-5)
    np.testing.assert_allclose(stds, 1.0, atol=1e-4)


def test_apply_preprocessing_all_known_steps_accepted():
    """All entries in KNOWN_PREPROC should not raise when dispatched."""
    x = _random_window()
    for step in KNOWN_PREPROC - {"euclidean_alignment"}:
        try:
            _apply_preprocessing(x.copy(), [step])
        except ValueError as e:
            if "unknown preprocessing step" in str(e):
                pytest.fail(f"KNOWN_PREPROC entry '{step}' unexpectedly raised: {e}")


def test_apply_preprocessing_accepts_parameterized_bandpass():
    pytest.importorskip("scipy.signal")
    x = _random_window()
    out = _apply_preprocessing(
        x.copy(),
        [{"name": "bandpass", "low_hz": 4.0, "high_hz": 45.0, "order": 4}],
    )
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# _fit_ea_on_windows
# ---------------------------------------------------------------------------

def test_fit_ea_on_windows_returns_none_without_ea():
    windows = [{"x": _random_window(), "y": 0, "subject_id": "s", "trial_id": "t",
                "cohort": "HC", "crop_id": 0, "window_start_sec": 0.0}]
    assert _fit_ea_on_windows(windows, ["zscore"]) is None


def test_fit_ea_on_windows_returns_matrix_with_ea():
    windows = [{"x": _random_window(i), "y": 0, "subject_id": "s", "trial_id": f"t{i}",
                "cohort": "HC", "crop_id": 0, "window_start_sec": 0.0}
               for i in range(4)]
    result = _fit_ea_on_windows(windows, ["euclidean_alignment", "zscore"])
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (30, 30)


# ---------------------------------------------------------------------------
# _make_fixed_crops
# ---------------------------------------------------------------------------


def test_make_fixed_crops_returns_non_overlapping_crops():
    x = np.arange(30 * 2500 * 5, dtype=np.float32).reshape(30, 2500 * 5)
    trials = [{
        "x": x,
        "y": 1,
        "subject_id": "S01",
        "trial_id": "S01_pos1",
        "cohort": "HC",
    }]
    crops = _make_fixed_crops(
        trials,
        source_trial_sec=50,
        window_sec=10,
        n_crops=5,
        preproc=[],
        skip_preproc=True,
    )
    assert len(crops) == 5
    assert [crop["crop_id"] for crop in crops] == [0, 1, 2, 3, 4]
    assert [crop["window_start_sec"] for crop in crops] == [0.0, 10.0, 20.0, 30.0, 40.0]
    np.testing.assert_array_equal(crops[1]["x"], x[:, 2500:5000])


def test_validate_fixed_crop_coverage_requires_source_duration():
    x = np.zeros((30, 2500 * 5), dtype=np.float32)
    trials = [{"x": x, "trial_id": "S01_pos1"}]
    with pytest.raises(ValueError, match="source_trial_sec=60"):
        _validate_fixed_crop_coverage(
            trials,
            source_trial_sec=60,
            window_sec=10,
            n_crops=5,
            split_name="test",
        )


# ---------------------------------------------------------------------------
# .mat loading and evidence manifests
# ---------------------------------------------------------------------------


def _write_hdf5_mat(path: Path, *, transpose: bool = False, samples_per_trial: int = 50) -> None:
    h5py = pytest.importorskip("h5py")
    total = samples_per_trial * 4
    neu = np.arange(30 * total, dtype=np.float32).reshape(30, total)
    pos = (neu + 1000.0).astype(np.float32)
    if transpose:
        neu = neu.T
        pos = pos.T
    with h5py.File(path, "w") as f:
        f.create_dataset("EEG_data_neu", data=neu)
        f.create_dataset("EEG_data_pos", data=pos)


def test_read_mat_accepts_channel_first_and_channel_last_hdf5(tmp_path):
    channel_first = tmp_path / "DEP001timedata.mat"
    channel_last = tmp_path / "HC001timedata.mat"
    _write_hdf5_mat(channel_first, transpose=False)
    _write_hdf5_mat(channel_last, transpose=True)

    assert _read_mat(channel_first, "EEG_data_neu").shape == (30, 200)
    assert _read_mat(channel_last, "EEG_data_pos").shape == (30, 200)


def test_load_mat_trials_reads_dep_hc_subjects_and_trials(tmp_path):
    _write_hdf5_mat(tmp_path / "DEP001timedata.mat", transpose=False)
    _write_hdf5_mat(tmp_path / "HC001timedata.mat", transpose=True)

    trials = _load_mat_trials(tmp_path)

    assert len(trials) == 16
    assert {trial["cohort"] for trial in trials} == {"DEP", "HC"}
    assert {trial["subject_id"] for trial in trials} == {"DEP001", "HC001"}
    assert {trial["y"] for trial in trials} == {0, 1}
    assert all(trial["x"].shape == (30, 50) for trial in trials)


def test_load_mat_trials_rejects_unknown_subject_prefix(tmp_path):
    _write_hdf5_mat(tmp_path / "BAD001timedata.mat", transpose=False)

    with pytest.raises(ValueError, match="expected DEP\\* or HC\\*"):
        _load_mat_trials(tmp_path)


def test_candidate_non_sliding_evidence_declares_fixed_five_crops(tmp_path):
    trials = []
    for subject_id, cohort in [("DEP001", "DEP"), ("HC001", "HC")]:
        for idx in range(8):
            trials.append({
                "x": np.zeros((30, 2500 * FIXED_CANDIDATE_CROPS), dtype=np.float32),
                "y": 1 if idx < 4 else 0,
                "subject_id": subject_id,
                "cohort": cohort,
                "trial_id": f"{subject_id}_t{idx}",
            })
    dataset_path, _ = _write_evidence_manifests(
        tmp_path / "run",
        route_data={
            "dataset_version": "train_v1",
            "split_id": "split1",
            "input_window_sec": 10,
        },
        active_split_id="split1",
        all_trials=trials,
        train_subjects={"DEP001"},
        val_subjects=set(),
        test_subjects={"HC001"},
        run_mode="candidate",
    )

    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    assert dataset["n_crops"] == FIXED_CANDIDATE_CROPS
    first = dataset["trial_index"][0]
    assert first["crop_ids"] == [0, 1, 2, 3, 4]
    assert first["window_start_secs"] == [0, 10, 20, 30, 40]


def test_run_real_classifier_route_candidate_fake_hdf5_is_test_only(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    import hust_bci_er.models.factory as model_factory
    import hust_bci_er.training.reproducibility as reproducibility

    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setattr(reproducibility, "PROCESS_START_PYTHONHASHSEED", "42")

    data_root = tmp_path / "data"
    data_root.mkdir()
    for idx in range(3):
        _write_hdf5_mat(data_root / f"DEP{idx:03d}timedata.mat", samples_per_trial=50)
        _write_hdf5_mat(data_root / f"HC{idx:03d}timedata.mat", samples_per_trial=50, transpose=True)

    route = tmp_path / "fake_real_route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: fake_real_route",
                "status: IDEA",
                "dataset_version: train_v1",
                "split_id: p1_seed42_fold0",
                "seed: 42",
                "input_window_sec: 0.04",
                "preprocessing: [zscore]",
                "features: []",
                "model:",
                "  name: eegnet",
                "adaptation: none",
                "training:",
                "  trainer: torch_classifier",
                "  job_adapter: torch_classifier",
                "  epochs: 1",
                "  batch_size: 8",
                "  optimizer:",
                "    name: sgd",
                "    lr: 0.01",
                "    weight_decay: 0.0",
                "  loss: cross_entropy",
                "inference:",
                "  top4: true",
                "  crop_policy: single",
                "evaluation:",
                "  protocol: p1_repeated_group_kfold",
                "  primary_metric: exact_single_crop_expected_BA",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class TinyClassifier(torch.nn.Module):
        def __init__(self, n_channels: int, n_times: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(n_channels * n_times, 2)

        def forward(self, x):
            return self.linear(x.flatten(1))

    captured_model_kwargs = {}

    def fake_build_model(_name, *, n_channels, n_times, n_classes, **kwargs):
        assert n_classes == 2
        captured_model_kwargs.update(kwargs)
        return TinyClassifier(n_channels, n_times)

    def fake_context(**kwargs):
        return SimpleNamespace(
            run_dir=kwargs["run_dir"],
            prediction_csv=kwargs["prediction_csv"],
        )

    def fake_manifest(context, *, metrics, command, score_matrix_csv):
        return {
            "audit_schema_version": 2,
            "route_id": "fake_real_route",
            "command": " ".join(command) if isinstance(command, list) else command,
            "primary_metric": "exact_single_crop_expected_BA",
            "metrics": dict(metrics),
            "prediction_csv": context.prediction_csv.name,
            "metric_inputs": {"score_matrix_csv": Path(score_matrix_csv).name},
        }

    monkeypatch.setattr(model_factory, "build_model", fake_build_model)
    monkeypatch.setattr(real_adapter, "prepare_run_manifest_context", fake_context)
    monkeypatch.setattr(real_adapter, "build_run_manifest_payload", fake_manifest)

    run_dir = tmp_path / "run"
    artifacts = run_real_classifier_route(
        route_config_path=route,
        run_dir=run_dir,
        run_mode="candidate",
        command=["python", "scripts/train_route.py", "--data-root", data_root.as_posix()],
        data_root=data_root,
        device="cpu",
    )

    manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))
    assert manifest["prediction_scope"] == "test_only"
    assert manifest["score_matrix_evidence"] == "genuine"

    split = yaml.safe_load(artifacts.split_manifest.read_text(encoding="utf-8"))
    test_subjects = set(split["test_subjects"])
    with artifacts.prediction_csv.open(newline="", encoding="utf-8") as f:
        prediction_subjects = {row["subject_id"] for row in csv.DictReader(f)}
    assert prediction_subjects == test_subjects

    dataset = yaml.safe_load(artifacts.dataset_manifest.read_text(encoding="utf-8"))
    assert dataset["n_crops"] == FIXED_CANDIDATE_CROPS
    assert dataset["raw_data_sources"]
    assert captured_model_kwargs == {}


def test_run_real_classifier_route_passes_model_kwargs_to_builder(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import hust_bci_er.models.factory as model_factory
    import hust_bci_er.training.reproducibility as reproducibility

    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setattr(reproducibility, "PROCESS_START_PYTHONHASHSEED", "42")

    data_root = tmp_path / "data"
    data_root.mkdir()
    for idx in range(3):
        _write_hdf5_mat(data_root / f"DEP{idx:03d}timedata.mat", samples_per_trial=50)
        _write_hdf5_mat(data_root / f"HC{idx:03d}timedata.mat", samples_per_trial=50, transpose=True)
    route = tmp_path / "fake_real_route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: fake_real_route",
                "status: IDEA",
                "dataset_version: train_v1",
                "split_id: p1_seed42_fold0",
                "seed: 42",
                "input_window_sec: 0.04",
                "preprocessing: [zscore]",
                "features: []",
                "model:",
                "  name: shallow_conv_net",
                "  n_filters_time: 24",
                "  drop_prob: 0.25",
                "adaptation: none",
                "training:",
                "  trainer: torch_classifier",
                "  job_adapter: torch_classifier",
                "  epochs: 1",
                "  batch_size: 8",
                "  optimizer:",
                "    name: adamw",
                "    lr: 0.001",
                "  loss: cross_entropy",
                "inference:",
                "  top4: true",
                "  crop_policy: single",
                "evaluation:",
                "  protocol: p1_repeated_group_kfold",
                "  primary_metric: exact_single_crop_expected_BA",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class TinyClassifier(torch.nn.Module):
        def __init__(self, n_channels: int, n_times: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(n_channels * n_times, 2)

        def forward(self, x):
            return self.linear(x.flatten(1))

    captured = {}

    def fake_build_model(_name, *, n_channels, n_times, n_classes, **kwargs):
        assert n_classes == 2
        captured.update(kwargs)
        return TinyClassifier(n_channels, n_times)

    def fake_context(**kwargs):
        return SimpleNamespace(
            run_dir=kwargs["run_dir"],
            prediction_csv=kwargs["prediction_csv"],
        )

    def fake_manifest(context, *, metrics, command, score_matrix_csv):
        return {
            "audit_schema_version": 2,
            "route_id": "fake_real_route",
            "command": " ".join(command) if isinstance(command, list) else command,
            "primary_metric": "exact_single_crop_expected_BA",
            "metrics": dict(metrics),
            "prediction_csv": context.prediction_csv.name,
            "metric_inputs": {"score_matrix_csv": Path(score_matrix_csv).name},
        }

    monkeypatch.setattr(model_factory, "build_model", fake_build_model)
    monkeypatch.setattr(real_adapter, "prepare_run_manifest_context", fake_context)
    monkeypatch.setattr(real_adapter, "build_run_manifest_payload", fake_manifest)

    run_real_classifier_route(
        route_config_path=route,
        run_dir=tmp_path / "run",
        run_mode="candidate",
        command=["python", "scripts/train_route.py", "--data-root", data_root.as_posix()],
        data_root=data_root,
        device="cpu",
    )

    assert captured == {"n_filters_time": 24, "drop_prob": 0.25}
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_kwargs"] == {"n_filters_time": 24, "drop_prob": 0.25}


def test_run_real_classifier_route_fixed_crop_augmentation_is_genuine(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import hust_bci_er.models.factory as model_factory
    import hust_bci_er.training.reproducibility as reproducibility

    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setattr(reproducibility, "PROCESS_START_PYTHONHASHSEED", "42")

    data_root = tmp_path / "data"
    data_root.mkdir()
    for idx in range(3):
        _write_hdf5_mat(data_root / f"DEP{idx:03d}timedata.mat", samples_per_trial=50)
        _write_hdf5_mat(data_root / f"HC{idx:03d}timedata.mat", samples_per_trial=50, transpose=True)

    route = tmp_path / "fixed_crop_route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: fixed_crop_route",
                "status: IDEA",
                "dataset_version: train_v1",
                "split_id: p1_seed42_fold0",
                "seed: 42",
                "input_window_sec: 0.04",
                "preprocessing: [zscore]",
                "features: []",
                "augmentation:",
                "  name: split_first_fixed_crops",
                "  split_first: true",
                "  source_trial_sec: 0.2",
                "  window_sec: 0.04",
                "  n_crops: 5",
                "  apply_to_splits: [train, val, test]",
                "  aggregate_to_trial:",
                "    method: mean_score",
                "    tie_break: mean_score",
                "model:",
                "  name: eegnet",
                "adaptation: none",
                "training:",
                "  trainer: torch_classifier",
                "  job_adapter: torch_classifier",
                "  epochs: 1",
                "  batch_size: 8",
                "  optimizer:",
                "    name: sgd",
                "    lr: 0.01",
                "    weight_decay: 0.0",
                "  loss: cross_entropy",
                "inference:",
                "  top4: true",
                "  crop_policy: exact_single_crop",
                "evaluation:",
                "  protocol: p1_repeated_group_kfold",
                "  primary_metric: exact_single_crop_expected_BA",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class TinyClassifier(torch.nn.Module):
        def __init__(self, n_channels: int, n_times: int) -> None:
            super().__init__()
            self.n_times = n_times
            self.linear = torch.nn.Linear(n_channels * n_times, 2)

        def forward(self, x):
            return self.linear(x.flatten(1))

    captured = {}

    def fake_build_model(_name, *, n_channels, n_times, n_classes, **kwargs):
        assert n_classes == 2
        captured["n_times"] = n_times
        return TinyClassifier(n_channels, n_times)

    def fake_context(**kwargs):
        return SimpleNamespace(
            run_dir=kwargs["run_dir"],
            prediction_csv=kwargs["prediction_csv"],
        )

    def fake_manifest(context, *, metrics, command, score_matrix_csv):
        return {
            "audit_schema_version": 2,
            "route_id": "fixed_crop_route",
            "command": " ".join(command) if isinstance(command, list) else command,
            "primary_metric": "exact_single_crop_expected_BA",
            "metrics": dict(metrics),
            "prediction_csv": context.prediction_csv.name,
            "metric_inputs": {"score_matrix_csv": Path(score_matrix_csv).name},
        }

    monkeypatch.setattr(model_factory, "build_model", fake_build_model)
    monkeypatch.setattr(real_adapter, "prepare_run_manifest_context", fake_context)
    monkeypatch.setattr(real_adapter, "build_run_manifest_payload", fake_manifest)

    artifacts = run_real_classifier_route(
        route_config_path=route,
        run_dir=tmp_path / "run",
        run_mode="candidate",
        command=["python", "scripts/train_route.py", "--data-root", data_root.as_posix()],
        data_root=data_root,
        device="cpu",
    )

    manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))
    dataset = yaml.safe_load(artifacts.dataset_manifest.read_text(encoding="utf-8"))
    assert captured["n_times"] == 10
    assert dataset["n_crops"] == 5
    assert manifest["score_matrix_evidence"] == "genuine"
