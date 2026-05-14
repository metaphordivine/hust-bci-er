import numpy as np
import pytest

from hust_bci_er.contracts.records import EEGTrial
from hust_bci_er.data.windowing import (
    SlidingWindowSpec,
    sliding_windows_for_trial,
    split_first_sliding_windows,
    window_slices,
    window_start_times,
)


def test_sliding_window_counts_match_common_eeg_setup():
    assert len(window_start_times(SlidingWindowSpec(source_trial_sec=50, window_sec=10, stride_sec=5))) == 9
    assert window_start_times(SlidingWindowSpec(source_trial_sec=10, window_sec=4, stride_sec=2)) == (0, 2, 4, 6)


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
