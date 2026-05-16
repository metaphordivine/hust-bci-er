"""Train-split-only EEG window augmentations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


TRAIN_ONLY_SPLITS = {"train"}


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else int(seed))


def gaussian_noise(x: np.ndarray, *, rng: np.random.Generator, std: float = 0.01) -> np.ndarray:
    if std < 0:
        raise ValueError("gaussian_noise.std must be non-negative")
    signal = np.asarray(x, dtype=np.float32)
    return (signal + rng.normal(0.0, std, size=signal.shape)).astype(np.float32)


def channel_dropout(x: np.ndarray, *, rng: np.random.Generator, p: float = 0.1) -> np.ndarray:
    if not 0.0 <= p < 1.0:
        raise ValueError("channel_dropout.p must be in [0, 1)")
    signal = np.asarray(x, dtype=np.float32).copy()
    mask = rng.random(signal.shape[0]) < p
    signal[mask, :] = 0.0
    return signal


def time_mask(x: np.ndarray, *, rng: np.random.Generator, max_width: int = 25) -> np.ndarray:
    if max_width <= 0:
        raise ValueError("time_mask.max_width must be positive")
    signal = np.asarray(x, dtype=np.float32).copy()
    width = int(rng.integers(1, min(max_width, signal.shape[-1]) + 1))
    start = int(rng.integers(0, signal.shape[-1] - width + 1))
    signal[:, start:start + width] = 0.0
    return signal


def time_shift(x: np.ndarray, *, rng: np.random.Generator, max_shift: int = 12) -> np.ndarray:
    if max_shift < 0:
        raise ValueError("time_shift.max_shift must be non-negative")
    signal = np.asarray(x, dtype=np.float32)
    if max_shift == 0:
        return signal.copy()
    shift = int(rng.integers(-max_shift, max_shift + 1))
    return np.roll(signal, shift=shift, axis=-1).astype(np.float32)


AUGMENTATION_TRANSFORMS = {
    "gaussian_noise": gaussian_noise,
    "channel_dropout": channel_dropout,
    "time_mask": time_mask,
    "time_shift": time_shift,
}


def normalize_transform_config(item: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise ValueError("augmentation transform must be a mapping")
    name = item.get("name")
    if name not in AUGMENTATION_TRANSFORMS:
        raise ValueError(f"unknown augmentation transform: {name}")
    splits = item.get("apply_to_splits", ["train"])
    if not isinstance(splits, Sequence) or isinstance(splits, (str, bytes)) or not splits:
        raise ValueError("augmentation transform apply_to_splits must be a non-empty sequence")
    split_set = {str(split) for split in splits}
    if split_set != TRAIN_ONLY_SPLITS:
        raise ValueError("augmentation transforms must be train-only: apply_to_splits must be [train]")
    nested_params = item.get("params")
    params = dict(nested_params) if isinstance(nested_params, Mapping) else {}
    params.update({key: value for key, value in item.items() if key not in {"name", "apply_to_splits", "params"}})
    return {"name": str(name), "apply_to_splits": ["train"], "params": params}


def transform_configs_from_route(route_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    augmentation = route_data.get("augmentation")
    if not isinstance(augmentation, Mapping):
        return []
    transforms = augmentation.get("transforms") or []
    if not isinstance(transforms, list):
        raise ValueError("augmentation.transforms must be a list")
    return [normalize_transform_config(item) for item in transforms]


def apply_transforms_to_windows(
    windows: list[dict[str, Any]],
    transform_configs: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    if split != "train" or not transform_configs:
        return windows
    rng = _rng(seed)
    augmented: list[dict[str, Any]] = []
    for window in windows:
        item = dict(window)
        x = np.asarray(item["x"], dtype=np.float32)
        for config in transform_configs:
            normalized = normalize_transform_config(config)
            fn = AUGMENTATION_TRANSFORMS[normalized["name"]]
            x = fn(x, rng=rng, **normalized["params"])
        item["x"] = x.astype(np.float32)
        augmented.append(item)
    return augmented
