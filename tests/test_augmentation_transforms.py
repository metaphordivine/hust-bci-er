import numpy as np
import pytest

from hust_bci_er.data.augmentations import (
    amplitude_scale,
    apply_transforms_to_windows,
    channel_dropout,
    channel_noise,
    gaussian_noise,
    normalize_transform_config,
    random_bandstop,
    smooth_time_mask,
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
        (amplitude_scale, {"scale_range": [0.9, 1.1]}),
        (channel_dropout, {"p": 0.25}),
        (channel_noise, {"max_channels": 2, "noise_std_ratio": 0.03}),
        (random_bandstop, {"sfreq": 32.0, "width_hz": 2.0, "freq_range": [4.0, 10.0]}),
        (smooth_time_mask, {"mask_ratio": 0.10}),
        (time_mask, {"max_width": 8}),
        (time_shift, {"max_shift": 4}),
    ]:
        y = fn(x, rng=rng, **kwargs)
        assert y.shape == x.shape
        assert y.dtype == np.float32


def test_transform_prob_zero_preserves_training_window():
    windows = [{"x": np.ones((4, 32), dtype=np.float32), "y": 1}]
    transforms = [{"name": "gaussian_noise", "std": 10.0, "prob": 0.0, "apply_to_splits": ["train"]}]

    train = apply_transforms_to_windows(windows, transforms, seed=42, split="train")

    np.testing.assert_allclose(train[0]["x"], windows[0]["x"])


def test_channel_dropout_can_exclude_frontal_channels():
    rng = np.random.default_rng(0)
    x = np.ones((30, 16), dtype=np.float32)

    y = channel_dropout(
        x,
        rng=rng,
        max_drop_channels=8,
        exclude_channels=["FP1", "FP2", "F7", "F8"],
    )

    np.testing.assert_array_equal(y[[0, 1, 2, 6], :], x[[0, 1, 2, 6], :])
    assert np.any(np.isclose(y.sum(axis=1), 0.0))


def test_channel_noise_can_exclude_frontal_channels():
    rng = np.random.default_rng(0)
    x = np.ones((30, 16), dtype=np.float32)

    y = channel_noise(
        x,
        rng=rng,
        max_channels=8,
        noise_std_ratio=0.1,
        exclude_channels=["FP1", "FP2", "F7", "F8"],
    )

    np.testing.assert_array_equal(y[[0, 1, 2, 6], :], x[[0, 1, 2, 6], :])
    assert y.shape == x.shape


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


def test_random_bandstop_reduces_target_sine_energy():
    class FixedBandRng:
        def uniform(self, *args, **kwargs):
            return 10.0

    sfreq = 100.0
    t = np.arange(100, dtype=np.float32) / sfreq
    x = np.sin(2.0 * np.pi * 10.0 * t).astype(np.float32)[None, :]
    y = random_bandstop(
        x,
        rng=FixedBandRng(),
        sfreq=sfreq,
        width_hz=2.0,
        freq_range=[10.0, 10.0],
        attenuation=0.0,
    )

    assert float(np.mean(y * y)) < float(np.mean(x * x)) * 0.2
