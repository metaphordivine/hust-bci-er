import numpy as np
import pytest

from hust_bci_er.data.augmentations import (
    apply_transforms_to_windows,
    channel_dropout,
    gaussian_noise,
    normalize_transform_config,
    time_mask,
    time_shift,
)


def test_train_only_transform_config_rejects_eval_splits():
    with pytest.raises(ValueError, match="train-only"):
        normalize_transform_config({"name": "gaussian_noise", "std": 0.01, "apply_to_splits": ["train", "val"]})


def test_window_transforms_preserve_shape_and_dtype():
    rng = np.random.default_rng(0)
    x = np.ones((4, 32), dtype=np.float32)
    for fn, kwargs in [
        (gaussian_noise, {"std": 0.01}),
        (channel_dropout, {"p": 0.25}),
        (time_mask, {"max_width": 8}),
        (time_shift, {"max_shift": 4}),
    ]:
        y = fn(x, rng=rng, **kwargs)
        assert y.shape == x.shape
        assert y.dtype == np.float32


def test_apply_transforms_only_changes_train_split():
    windows = [{"x": np.ones((4, 32), dtype=np.float32), "y": 1}]
    transforms = [normalize_transform_config({"name": "gaussian_noise", "std": 1.0, "apply_to_splits": ["train"]})]

    train = apply_transforms_to_windows(windows, transforms, seed=42, split="train")
    val = apply_transforms_to_windows(windows, transforms, seed=42, split="val")

    assert not np.allclose(train[0]["x"], windows[0]["x"])
    assert np.allclose(val[0]["x"], windows[0]["x"])
    assert np.allclose(windows[0]["x"], 1.0)


def test_apply_transform_chain_is_order_independent_with_window_metadata():
    windows = [
        {
            "subject_id": "S01",
            "trial_id": "t1",
            "crop_id": 0,
            "window_start_sec": 0.0,
            "x": np.ones((4, 32), dtype=np.float32),
        },
        {
            "subject_id": "S02",
            "trial_id": "t2",
            "crop_id": 1,
            "window_start_sec": 10.0,
            "x": np.ones((4, 32), dtype=np.float32) * 2.0,
        },
    ]
    transforms = [
        {"name": "gaussian_noise", "std": 0.01, "apply_to_splits": ["train"]},
        {"name": "time_mask", "max_width": 4, "apply_to_splits": ["train"]},
    ]

    forward = apply_transforms_to_windows(windows, transforms, seed=7, split="train")
    reversed_out = apply_transforms_to_windows(list(reversed(windows)), transforms, seed=7, split="train")

    by_trial = {item["trial_id"]: item["x"] for item in forward}
    by_trial_reversed = {item["trial_id"]: item["x"] for item in reversed_out}
    for trial_id in by_trial:
        np.testing.assert_allclose(by_trial[trial_id], by_trial_reversed[trial_id])


def test_time_shift_uses_zero_padding_not_wraparound():
    class FixedShiftRng:
        def integers(self, *args, **kwargs):
            return 1

    x = np.arange(8, dtype=np.float32).reshape(1, 8)
    y = time_shift(x, rng=FixedShiftRng(), max_shift=2)
    np.testing.assert_array_equal(y, np.array([[0, 0, 1, 2, 3, 4, 5, 6]], dtype=np.float32))
