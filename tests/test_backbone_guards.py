import pytest

torch = pytest.importorskip("torch")

from hust_bci_er.models.backbones.cbramod import CBraMod
from hust_bci_er.models.backbones.fbstcnet import (
    FBSTCNet,
    Cheby2FilterBank,
    FFTRectangularFilterBank,
)
from hust_bci_er.models.factory import build_model


# ---------------------------------------------------------------------------
# CBraMod guards
# ---------------------------------------------------------------------------

def test_cbramod_rejects_patch_embedding_dim_mismatch():
    with pytest.raises(ValueError, match="equal d_model"):
        CBraMod(
            n_chans=30,
            n_outputs=2,
            n_times=256,
            patch_size=128,
            d_model=200,
        )


# ---------------------------------------------------------------------------
# FBSTCNet band-count guard
# ---------------------------------------------------------------------------

def test_fbstcnet_rejects_mismatched_band_count():
    with pytest.raises(ValueError, match="n_bands must match len\\(bands\\)"):
        FBSTCNet(
            n_chans=30,
            n_outputs=2,
            n_times=256,
            sfreq=200.0,
            n_bands=12,
            bands=[(4.0, 8.0), (8.0, 12.0)],
        )


# ---------------------------------------------------------------------------
# FBSTCNet filterbank_type and defaults
# ---------------------------------------------------------------------------

def test_fbstcnet_default_filterbank_is_cheby2():
    model = FBSTCNet(n_chans=30, n_outputs=2, n_times=256, sfreq=200.0)
    assert model.filterbank_type == "cheby2"
    assert isinstance(model.filterbank, Cheby2FilterBank)
    assert model.filterbank_params["filterbank_type"] == "cheby2"


def test_fbstcnet_explicit_cheby2():
    model = FBSTCNet(
        n_chans=30, n_outputs=2, n_times=256, sfreq=200.0,
        filterbank_type="cheby2",
    )
    assert isinstance(model.filterbank, Cheby2FilterBank)


def test_fbstcnet_fft_rectangular_ablation():
    model = FBSTCNet(
        n_chans=30, n_outputs=2, n_times=256, sfreq=200.0,
        filterbank_type="fft_rectangular",
    )
    assert model.filterbank_type == "fft_rectangular"
    assert isinstance(model.filterbank, FFTRectangularFilterBank)
    assert model.filterbank_params["filterbank_type"] == "fft_rectangular"


def test_fbstcnet_rejects_unknown_filterbank_type():
    with pytest.raises(ValueError, match="unknown filterbank_type"):
        FBSTCNet(
            n_chans=30, n_outputs=2, n_times=256, sfreq=200.0,
            filterbank_type="butterworth",
        )


def test_fbstcnet_cheby2_records_filter_metadata():
    model = FBSTCNet(n_chans=30, n_outputs=2, n_times=256, sfreq=200.0)
    params = model.filterbank_params
    assert params["filterbank_type"] == "cheby2"
    assert params["sfreq"] == 200.0
    assert len(params["bands"]) == 12
    assert params["bands"][0] == (4.0, 8.0)
    assert params["bands"][-1] == (48.0, 52.0)
    assert params["order"] == 4
    assert params["stopband_ripple_db"] == 30.0
    assert params["transition_bw_hz"] == 2.0


# ---------------------------------------------------------------------------
# Filterbank output shapes
# ---------------------------------------------------------------------------

def test_cheby2_filterbank_output_shape():
    fb = Cheby2FilterBank(sfreq=200.0)
    fb.eval()
    with torch.no_grad():
        out = fb(torch.randn(2, 30, 256))
    assert tuple(out.shape) == (2, 12, 30, 256)


def test_fft_rectangular_filterbank_output_shape():
    fb = FFTRectangularFilterBank(sfreq=200.0)
    fb.eval()
    with torch.no_grad():
        out = fb(torch.randn(2, 30, 256))
    assert tuple(out.shape) == (2, 12, 30, 256)


def test_cheby2_filterbank_output_is_finite():
    fb = Cheby2FilterBank(sfreq=200.0)
    fb.eval()
    with torch.no_grad():
        out = fb(torch.randn(2, 30, 256))
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# build_model passthrough
# ---------------------------------------------------------------------------

def test_build_model_passes_filterbank_type_to_fbstcnet():
    model = build_model(
        "fbstcnet", n_channels=30, n_times=256, n_classes=2,
        filterbank_type="cheby2",
    )
    assert model.filterbank_type == "cheby2"

    model2 = build_model(
        "fbstcnet", n_channels=30, n_times=256, n_classes=2,
        filterbank_type="fft_rectangular",
    )
    assert model2.filterbank_type == "fft_rectangular"


def test_build_model_passes_cheby2_params_to_fbstcnet():
    model = build_model(
        "fbstcnet", n_channels=30, n_times=256, n_classes=2,
        filter_order=6,
        stopband_ripple=40.0,
        transition_bandwidth=1.5,
    )
    params = model.filterbank_params
    assert params["order"] == 6
    assert params["stopband_ripple_db"] == 40.0
    assert params["transition_bw_hz"] == 1.5
