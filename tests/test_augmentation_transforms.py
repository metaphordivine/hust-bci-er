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
