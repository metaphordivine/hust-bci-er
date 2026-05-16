import numpy as np
import pytest

from hust_bci_er.features.bandpower import bandpower_features
from hust_bci_er.features.connectivity import channel_correlation_matrix, connectivity_features
from hust_bci_er.features.differential_entropy import differential_entropy_features
from hust_bci_er.features.hjorth import hjorth_features
from hust_bci_er.features.riemannian import TangentSpaceProjector, regularized_covariance, tangent_space_features
from hust_bci_er.preprocessing.filtering import bandpass_filter
from hust_bci_er.preprocessing.normalization import zscore_per_channel
from hust_bci_er.preprocessing.whitening import channel_whiten


def test_preprocessing_shapes():
    x = np.random.default_rng(0).normal(size=(30, 250))
    assert zscore_per_channel(x).shape == x.shape
    assert channel_whiten(x, eps=1e-3).shape == x.shape


def test_feature_shapes():
    x = np.random.default_rng(1).normal(size=(30, 250))
    assert bandpower_features(x, sfreq=250).shape == (30, 5)
    assert connectivity_features(x).shape == (30 * 29 // 2,)
    assert differential_entropy_features(x, sfreq=250).shape == (30, 5)
    assert hjorth_features(x).shape == (30, 3)


def test_bandpass_filter_shape_and_finiteness():
    pytest.importorskip("scipy.signal")
    x = np.random.default_rng(2).normal(size=(30, 1000))
    y = bandpass_filter(x, sfreq=250.0, low_hz=4.0, high_hz=45.0, order=4)
    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert np.isfinite(y).all()


def test_bandpass_filter_short_signal_warns_about_padlen_fallback():
    pytest.importorskip("scipy.signal")
    x = np.random.default_rng(22).normal(size=(4, 20))
    with pytest.warns(RuntimeWarning, match="padlen=0"):
        y = bandpass_filter(x, sfreq=250.0, low_hz=4.0, high_hz=45.0, order=4)
    assert y.shape == x.shape


def test_connectivity_correlation_is_symmetric():
    x = np.random.default_rng(3).normal(size=(6, 200))
    corr = channel_correlation_matrix(x)
    assert corr.shape == (6, 6)
    assert np.allclose(corr, corr.T, atol=1e-6)
    assert np.allclose(np.diag(corr), 1.0)


def test_riemannian_tangent_features_are_vectorized_spd_features():
    x = np.random.default_rng(4).normal(size=(6, 200))
    cov = regularized_covariance(x, eps=1e-4, shrinkage_alpha=0.1)
    assert cov.shape == (6, 6)
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    feat = tangent_space_features(x, reference_covariance=cov, eps=1e-4)
    assert feat.shape == (6 * 7 // 2,)
    assert np.isfinite(feat).all()


def test_tangent_space_projector_fit_transform_shape():
    rng = np.random.default_rng(5)
    windows = [rng.normal(size=(6, 200)) for _ in range(4)]
    features = TangentSpaceProjector(eps=1e-4, shrinkage_alpha=0.1).fit_transform(windows)
    assert features.shape == (4, 6 * 7 // 2)
    assert np.isfinite(features).all()


def test_tangent_space_projector_transform_requires_fit():
    with pytest.raises(ValueError, match="must be fit"):
        TangentSpaceProjector().transform([np.zeros((6, 20), dtype=np.float32)])

