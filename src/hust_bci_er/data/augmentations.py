"""Train-split-only EEG window augmentations."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from hust_bci_er.models.eeg_montage import HUST_30_A2_CHANNELS, indices_for_channel_names


TRAIN_ONLY_SPLITS = {"train"}


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else int(seed))


def gaussian_noise(
    x: np.ndarray,
    *,
    rng: np.random.Generator,
    std: float = 0.01,
    std_ratio: float | None = None,
) -> np.ndarray:
    if std < 0:
        raise ValueError("gaussian_noise.std must be non-negative")
    signal = np.asarray(x, dtype=np.float32)
    if std_ratio is not None:
        if std_ratio < 0:
            raise ValueError("gaussian_noise.std_ratio must be non-negative")
        std = float(std_ratio) * float(np.std(signal))
    return (signal + rng.normal(0.0, float(std), size=signal.shape)).astype(np.float32)


def amplitude_scale(
    x: np.ndarray,
    *,
    rng: np.random.Generator,
    scale_range: Sequence[float] = (0.9, 1.1),
) -> np.ndarray:
    if not isinstance(scale_range, Sequence) or len(scale_range) != 2:
        raise ValueError("amplitude_scale.scale_range must contain [low, high]")
    low = float(scale_range[0])
    high = float(scale_range[1])
    if low <= 0 or high <= 0 or low > high:
        raise ValueError("amplitude_scale.scale_range must be positive and sorted")
    signal = np.asarray(x, dtype=np.float32)
    return (signal * float(rng.uniform(low, high))).astype(np.float32)


def channel_dropout(
    x: np.ndarray,
    *,
    rng: np.random.Generator,
    p: float = 0.1,
    max_drop_channels: int | None = None,
    exclude_channels: Sequence[str] | None = None,
    channel_names: Sequence[str] = HUST_30_A2_CHANNELS,
) -> np.ndarray:
    if not 0.0 <= p < 1.0:
        raise ValueError("channel_dropout.p must be in [0, 1)")
    signal = np.asarray(x, dtype=np.float32).copy()
    if max_drop_channels is not None:
        if max_drop_channels <= 0:
            raise ValueError("channel_dropout.max_drop_channels must be positive")
        excluded = set(indices_for_channel_names(channel_names, exclude_channels or ()))
        eligible = [idx for idx in range(signal.shape[0]) if idx not in excluded]
        if not eligible:
            return signal
        n_drop = int(rng.integers(1, min(int(max_drop_channels), len(eligible)) + 1))
        selected = rng.choice(np.asarray(eligible, dtype=np.int64), size=n_drop, replace=False)
        signal[selected, :] = 0.0
        return signal.astype(np.float32)
    mask = rng.random(signal.shape[0]) < float(p)
    signal[mask, :] = 0.0
    return signal


def channel_noise(
    x: np.ndarray,
    *,
    rng: np.random.Generator,
    max_channels: int = 2,
    noise_std_ratio: float = 0.03,
    exclude_channels: Sequence[str] | None = None,
    channel_names: Sequence[str] = HUST_30_A2_CHANNELS,
) -> np.ndarray:
    if max_channels <= 0:
        raise ValueError("channel_noise.max_channels must be positive")
    if noise_std_ratio < 0:
        raise ValueError("channel_noise.noise_std_ratio must be non-negative")
    signal = np.asarray(x, dtype=np.float32).copy()
    excluded = set(indices_for_channel_names(channel_names, exclude_channels or ()))
    eligible = [idx for idx in range(signal.shape[0]) if idx not in excluded]
    if not eligible or noise_std_ratio == 0:
        return signal
    n_channels = int(rng.integers(1, min(int(max_channels), len(eligible)) + 1))
    selected = rng.choice(np.asarray(eligible, dtype=np.int64), size=n_channels, replace=False)
    noise_std = float(noise_std_ratio) * max(float(np.std(signal)), 1e-12)
    signal[selected, :] += rng.normal(0.0, noise_std, size=(n_channels, signal.shape[-1])).astype(np.float32)
    return signal.astype(np.float32)


def _mask_width(n_times: int, *, max_width: int | None = None, mask_ratio: float | None = None) -> int:
    if mask_ratio is not None:
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        return max(1, int(round(float(mask_ratio) * n_times)))
    if max_width is None or max_width <= 0:
        raise ValueError("max_width must be positive")
    return int(max_width)


def time_mask(
    x: np.ndarray,
    *,
    rng: np.random.Generator,
    max_width: int = 25,
    mask_ratio: float | None = None,
) -> np.ndarray:
    signal = np.asarray(x, dtype=np.float32).copy()
    width_limit = _mask_width(signal.shape[-1], max_width=max_width, mask_ratio=mask_ratio)
    width = int(rng.integers(1, min(width_limit, signal.shape[-1]) + 1))
    start = int(rng.integers(0, signal.shape[-1] - width + 1))
    signal[:, start:start + width] = 0.0
    return signal


def smooth_time_mask(
    x: np.ndarray,
    *,
    rng: np.random.Generator,
    mask_ratio: float = 0.05,
    attenuation: float = 0.0,
    edge_ratio: float = 0.2,
) -> np.ndarray:
    if not 0.0 <= attenuation <= 1.0:
        raise ValueError("smooth_time_mask.attenuation must be in [0, 1]")
    if not 0.0 <= edge_ratio < 0.5:
        raise ValueError("smooth_time_mask.edge_ratio must be in [0, 0.5)")
    signal = np.asarray(x, dtype=np.float32).copy()
    width = min(_mask_width(signal.shape[-1], mask_ratio=mask_ratio), signal.shape[-1])
    start = int(rng.integers(0, signal.shape[-1] - width + 1))
    local = np.full(width, float(attenuation), dtype=np.float32)
    edge = int(round(width * float(edge_ratio)))
    if edge > 0 and width > 2 * edge:
        ramp_down = np.linspace(1.0, float(attenuation), edge, endpoint=False, dtype=np.float32)
        ramp_up = np.linspace(float(attenuation), 1.0, edge, endpoint=False, dtype=np.float32)
        local[:edge] = ramp_down
        local[-edge:] = ramp_up
    signal[:, start:start + width] *= local[None, :]
    return signal.astype(np.float32)


def time_shift(x: np.ndarray, *, rng: np.random.Generator, max_shift: int = 12) -> np.ndarray:
    if max_shift < 0:
        raise ValueError("time_shift.max_shift must be non-negative")
    signal = np.asarray(x, dtype=np.float32)
    if max_shift == 0:
        return signal.copy()
    shift = int(rng.integers(-max_shift, max_shift + 1))
    shifted = np.zeros_like(signal)
    if shift > 0:
        shifted[:, shift:] = signal[:, :-shift]
    elif shift < 0:
        shifted[:, :shift] = signal[:, -shift:]
    else:
        shifted[:, :] = signal
    return shifted.astype(np.float32)


def random_bandstop(
    x: np.ndarray,
    *,
    rng: np.random.Generator,
    sfreq: float = 250.0,
    width_hz: float = 1.0,
    freq_range: Sequence[float] = (4.0, 40.0),
    attenuation: float = 0.0,
) -> np.ndarray:
    if sfreq <= 0:
        raise ValueError("random_bandstop.sfreq must be positive")
    if width_hz <= 0:
        raise ValueError("random_bandstop.width_hz must be positive")
    if not 0.0 <= attenuation <= 1.0:
        raise ValueError("random_bandstop.attenuation must be in [0, 1]")
    if not isinstance(freq_range, Sequence) or len(freq_range) != 2:
        raise ValueError("random_bandstop.freq_range must contain [low, high]")
    low = float(freq_range[0])
    high = float(freq_range[1])
    if low < 0 or high < low:
        raise ValueError("random_bandstop.freq_range must be non-negative and sorted")
    nyquist = float(sfreq) / 2.0
    if high >= nyquist:
        raise ValueError("random_bandstop.freq_range high must be below Nyquist")

    signal = np.asarray(x, dtype=np.float32)
    n_times = signal.shape[-1]
    if n_times < 2:
        return signal.copy()
    center = float(rng.uniform(low, high))
    half_width = float(width_hz) / 2.0
    freqs = np.fft.rfftfreq(n_times, d=1.0 / float(sfreq))
    mask = (freqs >= center - half_width) & (freqs <= center + half_width)
    if not np.any(mask):
        return signal.copy()
    spectrum = np.fft.rfft(signal.astype(np.float64), axis=-1)
    spectrum[..., mask] *= float(attenuation)
    return np.fft.irfft(spectrum, n=n_times, axis=-1).astype(np.float32)


def band_amplitude_scale(
    x: np.ndarray,
    *,
    rng: np.random.Generator,
    sfreq: float = 250.0,
    bands: Mapping[str, Sequence[float]] | None = None,
    scale_range: Sequence[float] = (0.9, 1.1),
) -> np.ndarray:
    if sfreq <= 0:
        raise ValueError("band_amplitude_scale.sfreq must be positive")
    if not isinstance(scale_range, Sequence) or len(scale_range) != 2:
        raise ValueError("band_amplitude_scale.scale_range must contain [low, high]")
    low_scale = float(scale_range[0])
    high_scale = float(scale_range[1])
    if low_scale <= 0 or high_scale <= 0 or low_scale > high_scale:
        raise ValueError("band_amplitude_scale.scale_range must be positive and sorted")
    band_map = bands or {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
    signal = np.asarray(x, dtype=np.float32)
    n_times = signal.shape[-1]
    if n_times < 2:
        return signal.copy()
    freqs = np.fft.rfftfreq(n_times, d=1.0 / float(sfreq))
    spectrum = np.fft.rfft(signal.astype(np.float64), axis=-1)
    for name, bounds in band_map.items():
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise ValueError(f"band_amplitude_scale.bands.{name} must contain [low, high]")
        low = float(bounds[0])
        high = float(bounds[1])
        if low < 0 or high <= low or high >= float(sfreq) / 2.0:
            raise ValueError(f"band_amplitude_scale.bands.{name} must be sorted below Nyquist")
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            spectrum[..., mask] *= float(rng.uniform(low_scale, high_scale))
    return np.fft.irfft(spectrum, n=n_times, axis=-1).astype(np.float32)


def dc_shift(
    x: np.ndarray,
    *,
    rng: np.random.Generator,
    offset_std_ratio: float = 0.02,
    per_channel: bool = True,
) -> np.ndarray:
    if offset_std_ratio < 0:
        raise ValueError("dc_shift.offset_std_ratio must be non-negative")
    signal = np.asarray(x, dtype=np.float32)
    if offset_std_ratio == 0:
        return signal.copy()
    scale = float(offset_std_ratio) * max(float(np.std(signal)), 1e-12)
    if per_channel:
        offsets = rng.normal(0.0, scale, size=(signal.shape[0], 1)).astype(np.float32)
    else:
        offsets = np.asarray(rng.normal(0.0, scale), dtype=np.float32)
    return (signal + offsets).astype(np.float32)


AUGMENTATION_TRANSFORMS = {
    "amplitude_scale": amplitude_scale,
    "band_amplitude_scale": band_amplitude_scale,
    "channel_noise": channel_noise,
    "dc_shift": dc_shift,
    "gaussian_noise": gaussian_noise,
    "channel_dropout": channel_dropout,
    "smooth_time_mask": smooth_time_mask,
    "random_bandstop": random_bandstop,
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


def _window_rng_seed(seed: int, window: Mapping[str, Any], ordinal: int) -> int:
    stable_keys = ("subject_id", "trial_id", "crop_id", "window_start_sec")
    stable_values = [str(window.get(key, "")) for key in stable_keys]
    fallback_ordinal = "" if any(stable_values) else str(ordinal)
    parts = [
        str(int(seed)),
        *stable_values,
        fallback_ordinal,
    ]
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def apply_transforms_to_windows(
    windows: list[dict[str, Any]],
    transform_configs: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    """Apply configured transforms to training windows only.

    Each window gets an independent deterministic RNG seeded from run seed and
    stable window metadata, so augmentation does not depend on list order when
    those metadata fields are present.
    """

    if split != "train" or not transform_configs:
        return windows
    normalized_configs = [normalize_transform_config(config) for config in transform_configs]
    augmented: list[dict[str, Any]] = []
    for ordinal, window in enumerate(windows):
        item = dict(window)
        x = np.asarray(item["x"], dtype=np.float32)
        rng = _rng(_window_rng_seed(seed, item, ordinal))
        for normalized in normalized_configs:
            fn = AUGMENTATION_TRANSFORMS[normalized["name"]]
            params = dict(normalized["params"])
            prob = float(params.pop("prob", 1.0))
            if not 0.0 <= prob <= 1.0:
                raise ValueError("augmentation transform prob must be in [0, 1]")
            if prob < 1.0 and float(rng.random()) >= prob:
                continue
            x = fn(x, rng=rng, **params)
        item["x"] = x.astype(np.float32)
        augmented.append(item)
    return augmented
