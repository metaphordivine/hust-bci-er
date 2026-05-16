"""Split-first window/crop augmentation for EEG trials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from hust_bci_er.contracts.records import EEGTrial, EEGWindow


@dataclass(frozen=True)
class SlidingWindowSpec:
    source_trial_sec: float
    window_sec: float
    stride_sec: float

    def __post_init__(self) -> None:
        for field_name in ("source_trial_sec", "window_sec", "stride_sec"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.window_sec > self.source_trial_sec:
            raise ValueError("window_sec must be <= source_trial_sec")

    @property
    def n_windows(self) -> int:
        return int(np.floor((self.source_trial_sec - self.window_sec) / self.stride_sec + 1e-9)) + 1


def window_start_times(spec: SlidingWindowSpec) -> tuple[float, ...]:
    return tuple(round(idx * spec.stride_sec, 10) for idx in range(spec.n_windows))


def sliding_window_spec_from_config(config: dict[str, Any]) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        source_trial_sec=float(config["source_trial_sec"]),
        window_sec=float(config["window_sec"]),
        stride_sec=float(config["stride_sec"]),
    )


def window_slices(n_times: int, spec: SlidingWindowSpec) -> tuple[tuple[int, int, float], ...]:
    if not isinstance(n_times, int) or n_times <= 0:
        raise ValueError("n_times must be a positive integer")

    window_len = int(round(spec.window_sec / spec.source_trial_sec * n_times))
    stride_len = int(round(spec.stride_sec / spec.source_trial_sec * n_times))
    if window_len <= 0 or stride_len <= 0:
        raise ValueError("window and stride must span at least one sample")

    slices: list[tuple[int, int, float]] = []
    for idx, start_sec in enumerate(window_start_times(spec)):
        start = idx * stride_len
        stop = start + window_len
        if stop > n_times:
            stop = n_times
            start = stop - window_len
        slices.append((start, stop, start_sec))
    return tuple(slices)


def sliding_windows_for_trial(trial: EEGTrial, spec: SlidingWindowSpec) -> tuple[EEGWindow, ...]:
    x = np.asarray(trial.x)
    if x.ndim != 2:
        raise ValueError("trial.x must be shaped [channels, time]")

    windows: list[EEGWindow] = []
    for crop_id, (start, stop, start_sec) in enumerate(window_slices(x.shape[1], spec)):
        windows.append(
            EEGWindow(
                x=x[:, start:stop],
                y=trial.y,
                subject_id=trial.subject_id,
                trial_id=trial.trial_id,
                crop_id=crop_id,
                window_start_sec=start_sec,
                split=trial.split,
            )
        )
    return tuple(windows)


def split_first_sliding_windows(trials: Iterable[EEGTrial], spec: SlidingWindowSpec) -> tuple[EEGWindow, ...]:
    seen_split: dict[tuple[str, str], str] = {}
    windows: list[EEGWindow] = []

    for trial in trials:
        key = (str(trial.subject_id), str(trial.trial_id))
        split = str(trial.split)
        if key in seen_split and seen_split[key] != split:
            raise ValueError(f"original trial crosses split before windowing: {key}")
        seen_split[key] = split
        windows.extend(sliding_windows_for_trial(trial, spec))
    return tuple(windows)


@dataclass(frozen=True)
class FixedCropSpec:
    source_trial_sec: float
    window_sec: float
    n_crops: int

    def __post_init__(self) -> None:
        for field_name in ("source_trial_sec", "window_sec"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if not isinstance(self.n_crops, int) or self.n_crops <= 0:
            raise ValueError("n_crops must be a positive integer")
        if self.window_sec * self.n_crops > self.source_trial_sec + 1e-9:
            raise ValueError("n_crops * window_sec must fit inside source_trial_sec")


def fixed_crop_start_times(spec: FixedCropSpec) -> tuple[float, ...]:
    return tuple(round(idx * spec.window_sec, 10) for idx in range(spec.n_crops))


def fixed_crop_spec_from_config(config: dict[str, Any]) -> FixedCropSpec:
    return FixedCropSpec(
        source_trial_sec=float(config["source_trial_sec"]),
        window_sec=float(config["window_sec"]),
        n_crops=int(config["n_crops"]),
    )


def fixed_crop_slices(n_times: int, spec: FixedCropSpec) -> tuple[tuple[int, int, float], ...]:
    if not isinstance(n_times, int) or n_times <= 0:
        raise ValueError("n_times must be a positive integer")

    window_len = int(round(spec.window_sec / spec.source_trial_sec * n_times))
    if window_len <= 0:
        raise ValueError("window must span at least one sample")

    slices: list[tuple[int, int, float]] = []
    for idx, start_sec in enumerate(fixed_crop_start_times(spec)):
        start = idx * window_len
        stop = start + window_len
        if stop > n_times:
            stop = n_times
            start = stop - window_len
        slices.append((start, stop, start_sec))
    return tuple(slices)


def fixed_crops_for_trial(trial: EEGTrial, spec: FixedCropSpec) -> tuple[EEGWindow, ...]:
    x = np.asarray(trial.x)
    if x.ndim != 2:
        raise ValueError("trial.x must be shaped [channels, time]")

    crops: list[EEGWindow] = []
    for crop_id, (start, stop, start_sec) in enumerate(fixed_crop_slices(x.shape[1], spec)):
        crops.append(
            EEGWindow(
                x=x[:, start:stop],
                y=trial.y,
                subject_id=trial.subject_id,
                trial_id=trial.trial_id,
                crop_id=crop_id,
                window_start_sec=start_sec,
                split=trial.split,
            )
        )
    return tuple(crops)


def split_first_fixed_crops(trials: Iterable[EEGTrial], spec: FixedCropSpec) -> tuple[EEGWindow, ...]:
    seen_split: dict[tuple[str, str], str] = {}
    crops: list[EEGWindow] = []

    for trial in trials:
        key = (str(trial.subject_id), str(trial.trial_id))
        split = str(trial.split)
        if key in seen_split and seen_split[key] != split:
            raise ValueError(f"original trial crosses split before fixed cropping: {key}")
        seen_split[key] = split
        crops.extend(fixed_crops_for_trial(trial, spec))
    return tuple(crops)
