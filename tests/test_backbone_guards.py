import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hust_bci_er.models.backbones.cbramod import CBraMod
from hust_bci_er.models.backbones.deformer_lite import EEGDeformerLite
from hust_bci_er.models.backbones.fbstcnet import (
    FBSTCNet,
    Cheby2FilterBank,
    FFTRectangularFilterBank,
    aggregate_branch_logits,
    aggregate_crop_logits,
)
from hust_bci_er.models.factory import build_model


# ---------------------------------------------------------------------------
# Deformer pooling guards
# ---------------------------------------------------------------------------

def test_deformer_attention_pooling_forward_shape():
    model = EEGDeformerLite(
        n_channels=30,
        n_times=256,
        n_classes=2,
        conv_channels=8,
        embedding_dim=16,
        transformer_depth=1,
        num_heads=4,
        pooling="attention",
        classifier_hidden_dim=8,
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 30, 256))
    assert tuple(out.shape) == (2, 2)


def test_deformer_rejects_unknown_pooling():
    with pytest.raises(ValueError, match="pooling must be"):
        EEGDeformerLite(n_channels=30, n_times=256, pooling="median")


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


def test_cbramod_rejects_longer_than_initialized_input():
    model = CBraMod(n_chans=30, n_outputs=2, n_times=256, patch_size=200)

    with pytest.raises(ValueError, match="expected <= 400 samples"):
        model(torch.randn(2, 30, 401))


def test_cbramod_mean_pooling_forward_shape():
    model = CBraMod(
        n_chans=30,
        n_outputs=2,
        n_times=400,
        patch_size=200,
        n_layer=1,
        classifier_pooling="mean",
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 30, 400))
    assert tuple(out.shape) == (2, 2)


def test_cbramod_rejects_unknown_classifier_pooling():
    with pytest.raises(ValueError, match="classifier_pooling"):
        CBraMod(n_chans=30, n_outputs=2, n_times=256, classifier_pooling="max")


# ---------------------------------------------------------------------------
# FBSTCNet band-count guard
# ---------------------------------------------------------------------------

def test_fbstcnet_rejects_mismatched_band_count():
    with pytest.raises(ValueError, match=r"n_bands must match len\(bands\)"):
        FBSTCNet(
            n_chans=30,
            n_outputs=2,
            n_times=256,
            sfreq=200.0,
            n_bands=12,
            bands=[(4.0, 8.0), (8.0, 12.0)],
        )


def test_fbstcnet_rejects_invalid_grouped_spatial_width():
    with pytest.raises(ValueError, match="F2 must be divisible by F1"):
        FBSTCNet(
            n_chans=30,
            n_outputs=2,
            n_times=256,
            sfreq=200.0,
            F1=32,
            F2=48,
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
    assert params["implementation"] == "magnitude_response_fft"
    assert params["phase_response"] == "discarded"
    assert params["sfreq"] == 200.0
    assert len(params["bands"]) == 12
    assert params["bands"][0] == (4.0, 8.0)
    assert params["bands"][-1] == (48.0, 52.0)
    assert params["order"] == 4
    assert params["stopband_ripple_db"] == 30.0
    assert params["transition_bw_hz"] == 2.0


def test_fbstcnet_transition_bandwidth_changes_cheby2_design():
    narrow = Cheby2FilterBank(sfreq=250.0, transition_bw=0.5)
    wide = Cheby2FilterBank(sfreq=250.0, transition_bw=2.0)
    assert not np.allclose(narrow._sos_list[0], wide._sos_list[0])


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


def test_fbstcnet_mixed_variant_fuses_branches_equally():
    power_crop_logits = torch.tensor([[[8.0, 0.0]]] * 4).transpose(0, 1)
    conn_crop_logits = torch.tensor([[[0.0, 8.0]]])
    crop_weighted = aggregate_crop_logits(torch.cat([power_crop_logits, conn_crop_logits], dim=1))
    branch_weighted = aggregate_branch_logits(
        [
            aggregate_crop_logits(power_crop_logits),
            aggregate_crop_logits(conn_crop_logits),
        ]
    )

    assert torch.argmax(crop_weighted, dim=-1).item() == 0
    probs = torch.exp(branch_weighted)
    assert probs[0, 0] == pytest.approx(probs[0, 1], abs=1e-4)


def test_fbstcnet_return_crop_logits_is_structured_for_mixed_variant():
    model = FBSTCNet(n_chans=30, n_outputs=2, n_times=256, sfreq=200.0, variant="M")
    model.eval()

    with torch.no_grad():
        result = model(torch.randn(2, 30, 256), return_crop_logits=True)

    assert set(result) == {"power", "connectivity", "combined"}
    assert result["combined"].shape[1] == result["power"].shape[1] + result["connectivity"].shape[1]


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


def test_build_model_passes_attention_pooling_to_deformer():
    model = build_model(
        "deformer_lite",
        n_channels=30,
        n_times=256,
        n_classes=2,
        pooling="attention",
        classifier_hidden_dim=8,
    )
    assert model.pooling == "attention"
    assert model.attention_pool is not None


def test_build_model_passes_classifier_pooling_to_cbramod():
    model = build_model(
        "cbramod",
        n_channels=30,
        n_times=400,
        n_classes=2,
        n_layer=1,
        classifier_pooling="mean",
    )
    assert model.classifier_pooling == "mean"


def test_build_model_passes_cbramod_patch_embedding_width():
    model = build_model(
        "cbramod",
        n_channels=30,
        n_times=250,
        n_classes=2,
        patch_size=125,
        d_model=250,
        n_layer=1,
        nhead=10,
        classifier_pooling="mean",
        conv_out_channels=50,
        group_norm_groups=10,
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 30, 250))
    assert tuple(out.shape) == (2, 2)
