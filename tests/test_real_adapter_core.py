"""Unit tests for real_adapter core helper functions.

Tests cover:
- _split_subjects: disjointness, capping, empty-train guard
- _build_score_matrix: 5-crop filling for single-crop and sliding-window policies
- _apply_preprocessing: unknown step rejection
- _fit_ea_on_windows: skips when euclidean_alignment not in preproc
"""
from __future__ import annotations

import numpy as np
import pytest

from hust_bci_er.training.real_adapter import (
    KNOWN_PREPROC,
    _apply_preprocessing,
    _build_score_matrix,
    _fit_ea_on_windows,
    _split_subjects,
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
            "y_true": y_true,
            "y_score": base_score + i * 0.01,
            "y_pred": 1,
        })
    return rows


def test_build_score_matrix_single_crop_produces_5_crops():
    rows = _make_window_rows("t1", n_windows=1, base_score=0.6)
    result = _build_score_matrix(rows, crop_policy="single", seed=42)
    assert len(result) == 1
    assert all(f"crop_{i}" in result[0] for i in range(5))


def test_build_score_matrix_single_crop_crops_near_base():
    rows = _make_window_rows("t1", n_windows=1, base_score=0.6)
    result = _build_score_matrix(rows, crop_policy="single", seed=42)
    for i in range(5):
        assert abs(float(result[0][f"crop_{i}"]) - 0.6) < 1e-4, (
            f"crop_{i} far from base score: {result[0][f'crop_{i}']}"
        )


def test_build_score_matrix_sliding_window_uses_real_scores():
    rows = _make_window_rows("t1", n_windows=5, base_score=0.5)
    result = _build_score_matrix(rows, crop_policy="sliding_window_vote", seed=42)
    assert len(result) == 1
    for i in range(5):
        expected = f"{0.5 + i * 0.01:.8f}"
        assert result[0][f"crop_{i}"] == expected


def test_build_score_matrix_preserves_y_true():
    rows = _make_window_rows("t1", n_windows=1, y_true=0)
    result = _build_score_matrix(rows, crop_policy="single", seed=42)
    assert result[0]["y_true"] == 0


def test_build_score_matrix_multiple_trials():
    rows = (
        _make_window_rows("t1", n_windows=1, subject_id="S01", base_score=0.8)
        + _make_window_rows("t2", n_windows=1, subject_id="S01", base_score=0.3)
    )
    result = _build_score_matrix(rows, crop_policy="single", seed=42)
    assert len(result) == 2
    trial_ids = {r["trial_id"] for r in result}
    assert trial_ids == {"t1", "t2"}


def test_build_score_matrix_seed_deterministic():
    rows = _make_window_rows("t1", n_windows=1, base_score=0.5)
    r1 = _build_score_matrix(rows, crop_policy="single", seed=99)
    r2 = _build_score_matrix(rows, crop_policy="single", seed=99)
    for i in range(5):
        assert r1[0][f"crop_{i}"] == r2[0][f"crop_{i}"]


def test_build_score_matrix_seed_independent_of_trial_id():
    """Crop scores must NOT depend on trial_id — only on seed and base score."""
    rows_a = _make_window_rows("trial_AAA", n_windows=1, base_score=0.5)
    rows_b = _make_window_rows("trial_ZZZ", n_windows=1, base_score=0.5)
    r_a = _build_score_matrix(rows_a, crop_policy="single", seed=42)
    r_b = _build_score_matrix(rows_b, crop_policy="single", seed=42)
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
