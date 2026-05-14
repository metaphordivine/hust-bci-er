import numpy as np

from hust_bci_er.features.bandpower import bandpower_features
from hust_bci_er.features.differential_entropy import differential_entropy_features
from hust_bci_er.features.hjorth import hjorth_features
from hust_bci_er.preprocessing.normalization import zscore_per_channel
from hust_bci_er.preprocessing.whitening import channel_whiten


def test_preprocessing_shapes():
    x = np.random.default_rng(0).normal(size=(30, 250))
    assert zscore_per_channel(x).shape == x.shape
    assert channel_whiten(x, eps=1e-3).shape == x.shape


def test_feature_shapes():
    x = np.random.default_rng(1).normal(size=(30, 250))
    assert bandpower_features(x, sfreq=250).shape == (30, 5)
    assert differential_entropy_features(x, sfreq=250).shape == (30, 5)
    assert hjorth_features(x).shape == (30, 3)

