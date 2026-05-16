from pathlib import Path

import numpy as np
import pytest
import yaml

from hust_bci_er.contracts.records import EEGTrial
from hust_bci_er.data.windowing import (
    FixedCropSpec,
    SlidingWindowSpec,
    fixed_crop_slices,
    fixed_crop_start_times,
    sliding_windows_for_trial,
    split_first_sliding_windows,
    window_slices,
    window_start_times,
)


def test_sliding_window_counts_match_common_eeg_setup():
    assert len(window_start_times(SlidingWindowSpec(source_trial_sec=50, window_sec=10, stride_sec=5))) == 9
    assert window_start_times(SlidingWindowSpec(source_trial_sec=10, window_sec=4, stride_sec=2)) == (0, 2, 4, 6)
    route = yaml.safe_load(Path("configs/routes/models/sliding_window_eegnet.yaml").read_text(encoding="utf-8"))
    assert len(window_start_times(SlidingWindowSpec(**{key: route["augmentation"][key] for key in ["source_trial_sec", "window_sec", "stride_sec"]}))) == 5


def test_sliding_windows_keep_original_trial_inside_one_split():
    trial = EEGTrial(
        x=np.arange(2 * 10).reshape(2, 10),
        y=1,
        subject_id="s1",
        trial_id="t1",
        split="train",
    )
    spec = SlidingWindowSpec(source_trial_sec=10, window_sec=4, stride_sec=2)

    windows = sliding_windows_for_trial(trial, spec)

    assert [window.crop_id for window in windows] == [0, 1, 2, 3]
    assert [window.window_start_sec for window in windows] == [0, 2, 4, 6]
    assert all(window.split == "train" for window in windows)
    assert all(window.trial_id == "t1" for window in windows)
    assert [window.x.shape for window in windows] == [(2, 4)] * 4
    assert windows[1].x.tolist() == np.arange(2 * 10).reshape(2, 10)[:, 2:6].tolist()


def test_split_first_guard_rejects_original_trial_across_splits():
    spec = SlidingWindowSpec(source_trial_sec=10, window_sec=4, stride_sec=2)
    trials = [
        EEGTrial(np.zeros((2, 10)), 0, "s1", "t1", "train"),
        EEGTrial(np.zeros((2, 10)), 0, "s1", "t1", "test"),
    ]

    with pytest.raises(ValueError, match="crosses split"):
        split_first_sliding_windows(trials, spec)


def test_window_slices_scale_with_sample_count():
    assert window_slices(100, SlidingWindowSpec(source_trial_sec=10, window_sec=4, stride_sec=2)) == (
        (0, 40, 0),
        (20, 60, 2),
        (40, 80, 4),
        (60, 100, 6),
    )


def test_window_slices_keep_constant_length_when_sample_count_is_odd():
    slices = window_slices(101, SlidingWindowSpec(source_trial_sec=10, window_sec=4, stride_sec=2))
    assert [stop - start for start, stop, _ in slices] == [40, 40, 40, 40]


def test_fixed_crop_slices_are_non_overlapping_and_do_not_left_shift():
    spec = FixedCropSpec(window_sec=10, n_crops=5)

    assert fixed_crop_start_times(spec) == (0, 10, 20, 30, 40)
    assert fixed_crop_slices(12500, spec, sampling_rate_hz=250) == (
        (0, 2500, 0),
        (2500, 5000, 10),
        (5000, 7500, 20),
        (7500, 10000, 30),
        (10000, 12500, 40),
    )


def test_fixed_crop_slices_reject_short_trials():
    with pytest.raises(ValueError, match="fixed crops require"):
        fixed_crop_slices(12499, FixedCropSpec(window_sec=10, n_crops=5), sampling_rate_hz=250)
