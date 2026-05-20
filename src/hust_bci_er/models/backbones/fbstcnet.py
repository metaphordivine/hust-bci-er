"""FBSTCNet: Filter-Bank Spatio-Temporal Convolutional Network.

Variants: P (power), C (connectivity), M (mixed).
Default filterbank: Chebyshev Type II magnitude-response FFT approximation
(12 non-overlapping 4 Hz bands, 4-52 Hz).
FFT rectangular filterbank available via filterbank_type="fft_rectangular" for ablation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def aggregate_crop_logits(crop_logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Aggregate crop-level logits to trial-level logits via mean in prob space."""
    prob = torch.softmax(crop_logits, dim=-1)
    prob_mean = prob.mean(dim=1)
    log_prob = torch.log(prob_mean.clamp_min(eps))
    return log_prob


def aggregate_branch_logits(branch_logits: list[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    """Fuse branch-level logits with equal branch weights in prob space."""
    if not branch_logits:
        raise ValueError("branch_logits must not be empty")
    if len(branch_logits) == 1:
        return branch_logits[0]
    prob = torch.stack([torch.softmax(logits, dim=-1) for logits in branch_logits], dim=0)
    prob_mean = prob.mean(dim=0)
    return torch.log(prob_mean.clamp_min(eps))


# ---------------------------------------------------------------------------
# Filterbank implementations
# ---------------------------------------------------------------------------

_DEFAULT_BANDS: list[tuple[float, float]] = [
    (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 20.0),
    (20.0, 24.0), (24.0, 28.0), (28.0, 32.0), (32.0, 36.0),
    (36.0, 40.0), (40.0, 44.0), (44.0, 48.0), (48.0, 52.0),
]


def _design_cheby2_sos(
    sfreq: float,
    bands: list[tuple[float, float]],
    order: int = 4,
    stopband_ripple: float = 30.0,
    transition_bw: float = 2.0,
) -> Any:
    """Design Chebyshev Type II bandpass SOS filters."""
    from scipy.signal import cheby2

    sos_list: list[Any] = []
    nyq = sfreq / 2.0
    for low, high in bands:
        ws = [(low - transition_bw) / nyq, (high + transition_bw) / nyq]
        ws = [max(ws[0], 1e-6), min(ws[1], 0.9999)]
        sos = cheby2(N=order, rs=stopband_ripple, Wn=ws, btype="bandpass", output="sos")
        sos_list.append(sos)
    return sos_list


def _compute_sos_freq_response(
    sos_list: list[Any], n_fft_bins: int, sfreq: float
) -> torch.Tensor:
    """Compute magnitude-only frequency responses from SOS filters."""
    import numpy as np
    from scipy.signal import sosfreqz

    B = len(sos_list)
    gains = torch.zeros(B, n_fft_bins, dtype=torch.float32)
    freqs_hz = torch.fft.rfftfreq((n_fft_bins - 1) * 2, d=1.0 / sfreq)[:n_fft_bins]

    for b_idx, sos in enumerate(sos_list):
        w, h = sosfreqz(sos, worN=n_fft_bins, fs=sfreq)
        if len(h) != n_fft_bins:
            w_full, h_full = sosfreqz(sos, worN=freqs_hz.numpy(), fs=sfreq, whole=False)
            h = h_full
        gains[b_idx] = torch.from_numpy(np.abs(h).astype(np.float32))
    return gains


class Cheby2FilterBank(nn.Module):
    """Cheby2 magnitude-response FFT approximation.

    The SOS design is used to derive a magnitude response. The complex phase
    response is intentionally not applied, so this is not time-domain SOS
    filtering.
    """

    def __init__(
        self,
        sfreq: float = 200.0,
        bands: list[tuple[float, float]] | None = None,
        order: int = 4,
        stopband_ripple: float = 30.0,
        transition_bw: float = 2.0,
    ) -> None:
        super().__init__()
        if bands is None:
            bands = list(_DEFAULT_BANDS)
        self.sfreq = sfreq
        self.bands = bands
        self.order = order
        self.stopband_ripple = stopband_ripple
        self.transition_bw = transition_bw

        self.metadata: dict[str, Any] = {
            "filterbank_type": "cheby2",
            "implementation": "magnitude_response_fft",
            "phase_response": "discarded",
            "sfreq": sfreq,
            "bands": [(float(low), float(high)) for low, high in bands],
            "order": order,
            "stopband_ripple_db": stopband_ripple,
            "transition_bw_hz": transition_bw,
        }

        sos_list = _design_cheby2_sos(
            sfreq=sfreq, bands=bands, order=order,
            stopband_ripple=stopband_ripple, transition_bw=transition_bw,
        )
        self._sos_list = sos_list
        self._gains: torch.Tensor | None = None
        self._gains_T: int = 0

    def _ensure_gains(self, T: int, device: torch.device) -> torch.Tensor:
        n_fft_bins = T // 2 + 1
        if self._gains is not None and self._gains_T == T:
            return self._gains.to(device=device)
        gains = _compute_sos_freq_response(self._sos_list, n_fft_bins, self.sfreq)
        self._gains = gains
        self._gains_T = T
        return gains.to(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        X = torch.fft.rfft(x, dim=-1)
        gains = self._ensure_gains(T, x.device)
        gains = gains.to(dtype=X.dtype)
        X_filt = X.unsqueeze(1) * gains.unsqueeze(0).unsqueeze(2)
        x_bands = torch.fft.irfft(X_filt, dim=-1, n=T)
        return x_bands


class FFTRectangularFilterBank(nn.Module):
    """FFT-based ideal bandpass filter bank (rectangular frequency mask).

    Ablation variant. Use ``filterbank_type: fft_rectangular`` to select.
    """

    def __init__(
        self,
        sfreq: float = 200.0,
        bands: list[tuple[float, float]] | None = None,
    ) -> None:
        super().__init__()
        if bands is None:
            bands = [(float(f), float(f + 4)) for f in range(4, 52, 4)]
        self.sfreq = sfreq
        self.bands = bands
        self.metadata: dict[str, Any] = {
            "filterbank_type": "fft_rectangular",
            "sfreq": sfreq,
            "bands": [(float(low), float(high)) for low, high in bands],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        X = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sfreq, device=x.device)
        outs: list[torch.Tensor] = []
        for low, high in self.bands:
            mask = (freqs >= low) & (freqs <= high)
            X_band = X * mask.to(dtype=X.dtype)
            x_band = torch.fft.irfft(X_band, dim=-1, n=T)
            outs.append(x_band)
        return torch.stack(outs, dim=1)


# ---------------------------------------------------------------------------
# STConv block and task heads
# ---------------------------------------------------------------------------


class FBSTC_STConvBlock(nn.Module):
    def __init__(
        self, n_bands: int, n_chans: int, sfreq: float,
        F1: int = 36, F2: int = 36, alpha: float = 0.125,
    ) -> None:
        super().__init__()
        kt = int(alpha * sfreq)
        self.temporal = nn.Conv2d(n_bands, F1, (1, kt), padding=(0, kt // 2), bias=False)
        self.spatial = nn.Conv2d(F1, F2, (n_chans, 1), groups=F1, bias=False)
        self.bn = nn.BatchNorm2d(F2)

    def forward(self, x_fb: torch.Tensor) -> torch.Tensor:
        z = self.temporal(x_fb)
        z = self.spatial(z)
        return self.bn(z)


class FBSTC_PowerHead(nn.Module):
    def __init__(
        self, F2: int, n_outputs: int, alpha1: int, alpha2: int,
        pool_size: int = 80, pool_stride: int = 5,
        drop_prob: float = 0.5, eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.pool = nn.AvgPool2d((1, pool_size), (1, pool_stride))
        self.drop = nn.Dropout(p=drop_prob)
        self.classifier = nn.Conv2d(F2, n_outputs, (1, alpha1), stride=(1, alpha2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z * z
        z = self.pool(z)
        z = torch.log(z.clamp_min(self.eps))
        z = self.drop(z)
        logits = self.classifier(z)
        return logits.squeeze(2).transpose(1, 2)


class FBSTC_ConnectivityHead(nn.Module):
    def __init__(
        self, F2: int, n_outputs: int, gamma: int, eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.F2 = F2
        self.gamma = gamma
        self.stride = max(gamma // 2, 1)
        self.eps = eps
        self.classifier = nn.Linear(F2 * F2, n_outputs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.squeeze(2)
        T = z.shape[-1]
        gamma = min(self.gamma, T)
        stride = max(gamma // 2, 1)
        windows = z.unfold(dimension=-1, size=gamma, step=stride)
        windows = windows.permute(0, 2, 1, 3).contiguous()
        corr = self._pearson(windows)
        feat = corr.flatten(start_dim=2)
        return self.classifier(feat)

    def _pearson(self, w: torch.Tensor) -> torch.Tensor:
        w = w - w.mean(dim=-1, keepdim=True)
        numerator = torch.matmul(w, w.transpose(-1, -2))
        power = torch.sum(w * w, dim=-1, keepdim=True)
        denom = torch.sqrt(power * power.transpose(-1, -2))
        return numerator / denom.clamp_min(self.eps)


class RiemannianBranchGate(nn.Module):
    """Predict branch weights from log-covariance tangent features."""

    def __init__(
        self,
        n_chans: int,
        n_branches: int,
        hidden_dim: int = 32,
        covariance_eps: float = 1e-3,
        shrinkage: float = 0.1,
        tangent_scale: float = math.sqrt(2.0),
        temperature: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_chans < 2:
            raise ValueError("RiemannianBranchGate requires at least two channels")
        if n_branches < 2:
            raise ValueError("RiemannianBranchGate requires at least two branches")
        if hidden_dim <= 0:
            raise ValueError("gate hidden_dim must be positive")
        if covariance_eps <= 0:
            raise ValueError("gate covariance_eps must be positive")
        if not 0.0 <= shrinkage < 1.0:
            raise ValueError("gate shrinkage must be in [0, 1)")
        if tangent_scale <= 0:
            raise ValueError("gate tangent_scale must be positive")
        if temperature <= 0:
            raise ValueError("gate temperature must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("gate dropout must be in [0, 1)")

        self.n_chans = int(n_chans)
        self.n_branches = int(n_branches)
        self.covariance_eps = float(covariance_eps)
        self.shrinkage = float(shrinkage)
        self.tangent_scale = float(tangent_scale)
        self.temperature = float(temperature)
        self.feature_dim = self.n_chans * (self.n_chans + 1) // 2

        rows, cols = torch.triu_indices(self.n_chans, self.n_chans)
        self.register_buffer("triu_rows", rows, persistent=False)
        self.register_buffer("triu_cols", cols, persistent=False)
        self.register_buffer("offdiag_mask", (rows != cols), persistent=False)

        self.feature_norm = nn.LayerNorm(self.feature_dim, elementwise_affine=False)
        self.gate = nn.Sequential(
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.n_branches),
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError("4D EEG input must be shaped [batch, 1, channels, times]")
            x = x.squeeze(1)
        if x.ndim != 3:
            raise ValueError("EEG input must be shaped [batch, channels, times]")
        if x.shape[1] != self.n_chans:
            raise ValueError(f"expected {self.n_chans} channels, got {x.shape[1]}")
        if x.shape[2] < 2:
            raise ValueError("EEG input must contain at least two time samples")
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        centered = x - x.mean(dim=-1, keepdim=True)
        denom = max(int(centered.shape[-1]) - 1, 1)
        cov = centered @ centered.transpose(-1, -2) / float(denom)
        eye = torch.eye(self.n_chans, dtype=cov.dtype, device=cov.device).expand(cov.shape[0], -1, -1)
        trace_mean = cov.diagonal(dim1=-2, dim2=-1).mean(dim=-1).view(-1, 1, 1)
        cov = (1.0 - self.shrinkage) * cov + self.shrinkage * trace_mean * eye
        cov = cov + self.covariance_eps * eye
        eigvals, eigvecs = torch.linalg.eigh(cov)
        log_diag = torch.log(torch.clamp(eigvals, min=self.covariance_eps))
        log_cov = (eigvecs * log_diag.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
        features = log_cov[:, self.triu_rows, self.triu_cols]
        features = torch.where(self.offdiag_mask.unsqueeze(0), features * self.tangent_scale, features)
        return self.feature_norm(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(self.extract_features(x)) / self.temperature
        return torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Top-level FBSTCNet
# ---------------------------------------------------------------------------


def _resolve_bands(
    bands: list[tuple[float, float]] | None, n_bands: int,
) -> list[tuple[float, float]]:
    if bands is None:
        return [(float(4 * idx), float(4 * idx + 4)) for idx in range(1, n_bands + 1)]
    if len(bands) != n_bands:
        raise ValueError(
            f"n_bands must match len(bands), got n_bands={n_bands} and len(bands)={len(bands)}"
        )
    return bands


def _build_filterbank(
    filterbank_type: str, sfreq: float, bands: list[tuple[float, float]],
    **fb_kwargs: Any,
) -> nn.Module:
    if filterbank_type == "cheby2":
        return Cheby2FilterBank(
            sfreq=sfreq, bands=bands,
            order=int(fb_kwargs.get("filter_order", 4)),
            stopband_ripple=float(fb_kwargs.get("stopband_ripple", 30.0)),
            transition_bw=float(fb_kwargs.get("transition_bandwidth", 2.0)),
        )
    if filterbank_type == "fft_rectangular":
        return FFTRectangularFilterBank(sfreq=sfreq, bands=bands)
    raise ValueError(
        f"unknown filterbank_type: {filterbank_type!r}, expected 'cheby2' or 'fft_rectangular'"
    )


class FBSTCNet(nn.Module):
    """FBSTCNet: Filter-Bank Spatio-Temporal Convolutional Network.

    Variants: P (power), C (connectivity), M (mixed).
    Default filterbank: Cheby2 magnitude-response FFT approximation.
    Use ``filterbank_type="fft_rectangular"`` for ablation.
    """

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        sfreq: float,
        n_times: int = 0,
        variant: str = "M",
        n_bands: int = 12,
        bands: list[tuple[float, float]] | None = None,
        filterbank_type: str = "cheby2",
        F1: int = 36,
        F2: int = 36,
        st_alpha: float = 0.125,
        alpha1: int = 25,
        alpha2: int = 5,
        pool_size: int = 80,
        pool_stride: int = 5,
        power_drop_prob: float = 0.5,
        gamma: int = 400,
        conn_eps: float = 1e-6,
        branch_fusion: str = "equal_probability_average",
        gate_hidden_dim: int = 32,
        gate_covariance_eps: float = 1e-3,
        gate_shrinkage: float = 0.1,
        gate_tangent_scale: float = math.sqrt(2.0),
        gate_temperature: float = 1.0,
        gate_dropout: float = 0.0,
        **filterbank_kwargs: Any,
    ) -> None:
        super().__init__()
        del n_times
        if variant not in ("P", "C", "M"):
            raise ValueError(f"variant must be P, C, or M, got {variant}")
        if F2 % F1 != 0:
            raise ValueError(f"F2 must be divisible by F1 for grouped spatial conv, got F1={F1}, F2={F2}")
        if branch_fusion not in {"equal_probability_average", "riemannian_gate"}:
            raise ValueError("branch_fusion must be equal_probability_average or riemannian_gate")
        if branch_fusion == "riemannian_gate" and variant != "M":
            raise ValueError("riemannian_gate branch_fusion requires mixed variant M")
        self.variant = variant
        self.sfreq = sfreq
        self.filterbank_type = filterbank_type
        self.branch_fusion = branch_fusion
        self.branch_names = ("power", "connectivity") if variant == "M" else (("power",) if variant == "P" else ("connectivity",))

        bands = _resolve_bands(bands, n_bands)

        self.filterbank = _build_filterbank(
            filterbank_type=filterbank_type, sfreq=sfreq, bands=bands,
            **filterbank_kwargs,
        )
        n_filter_bands = len(self.filterbank.bands)

        if variant in ("P", "M"):
            self.st_p = FBSTC_STConvBlock(
                n_bands=n_filter_bands, n_chans=n_chans, sfreq=sfreq,
                F1=F1, F2=F2, alpha=st_alpha,
            )
            self.power_head = FBSTC_PowerHead(
                F2=F2, n_outputs=n_outputs, alpha1=alpha1, alpha2=alpha2,
                pool_size=pool_size, pool_stride=pool_stride,
                drop_prob=power_drop_prob,
            )

        if variant in ("C", "M"):
            self.st_c = FBSTC_STConvBlock(
                n_bands=n_filter_bands, n_chans=n_chans, sfreq=sfreq,
                F1=F1, F2=F2, alpha=st_alpha,
            )
            self.conn_head = FBSTC_ConnectivityHead(
                F2=F2, n_outputs=n_outputs, gamma=gamma, eps=conn_eps,
            )

        if self.branch_fusion == "riemannian_gate":
            self.branch_gate = RiemannianBranchGate(
                n_chans=n_chans,
                n_branches=len(self.branch_names),
                hidden_dim=gate_hidden_dim,
                covariance_eps=gate_covariance_eps,
                shrinkage=gate_shrinkage,
                tangent_scale=gate_tangent_scale,
                temperature=gate_temperature,
                dropout=gate_dropout,
            )

    @property
    def filterbank_params(self) -> dict[str, Any]:
        if hasattr(self.filterbank, "metadata"):
            return dict(self.filterbank.metadata)
        return {"filterbank_type": self.filterbank_type}

    def forward(self, x: torch.Tensor, return_crop_logits: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        if x.dim() == 4:
            x = x.squeeze(1)
        x_fb = self.filterbank(x)
        branch_crop_logits: dict[str, torch.Tensor] = {}
        if self.variant in ("P", "M"):
            branch_crop_logits["power"] = self.power_head(self.st_p(x_fb))
        if self.variant in ("C", "M"):
            branch_crop_logits["connectivity"] = self.conn_head(self.st_c(x_fb))
        crop_logits = torch.cat([branch_crop_logits[name] for name in self.branch_names], dim=1)
        if return_crop_logits:
            result = {**branch_crop_logits, "combined": crop_logits}
            if self.branch_fusion == "riemannian_gate":
                result["branch_weights"] = self.branch_gate(x)
            return result
        branch_logits = [aggregate_crop_logits(branch_crop_logits[name]) for name in self.branch_names]
        if self.branch_fusion == "riemannian_gate":
            branch_weights = self.branch_gate(x)
            branch_probs = torch.stack([torch.softmax(logits, dim=-1) for logits in branch_logits], dim=1)
            fused_probs = (branch_weights.unsqueeze(-1) * branch_probs).sum(dim=1)
            return torch.log(fused_probs.clamp_min(1e-6))
        return aggregate_branch_logits(branch_logits)


def build_fbstcnet(
    n_channels: int, n_times: int, n_classes: int = 2, **kwargs: object
) -> FBSTCNet:
    filterbank_key_names = {"filter_order", "stopband_ripple", "transition_bandwidth"}
    filterbank_kwargs = {k: v for k, v in kwargs.items() if k in filterbank_key_names}
    model_kwargs = {k: v for k, v in kwargs.items() if k not in filterbank_key_names}
    return FBSTCNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        sfreq=float(model_kwargs.get("sfreq", 200.0)),
        variant=str(model_kwargs.get("variant", "M")),
        filterbank_type=str(model_kwargs.get("filterbank_type", "cheby2")),
        n_bands=int(model_kwargs.get("n_bands", 12)),
        bands=model_kwargs.get("bands", None),
        F1=int(model_kwargs.get("F1", 36)),
        F2=int(model_kwargs.get("F2", 36)),
        st_alpha=float(model_kwargs.get("st_alpha", 0.125)),
        alpha1=int(model_kwargs.get("alpha1", 25)),
        alpha2=int(model_kwargs.get("alpha2", 5)),
        pool_size=int(model_kwargs.get("pool_size", 80)),
        pool_stride=int(model_kwargs.get("pool_stride", 5)),
        power_drop_prob=float(model_kwargs.get("power_drop_prob", 0.5)),
        gamma=int(model_kwargs.get("gamma", 400)),
        conn_eps=float(model_kwargs.get("conn_eps", 1e-6)),
        branch_fusion=str(model_kwargs.get("branch_fusion", "equal_probability_average")),
        gate_hidden_dim=int(model_kwargs.get("gate_hidden_dim", 32)),
        gate_covariance_eps=float(model_kwargs.get("gate_covariance_eps", 1e-3)),
        gate_shrinkage=float(model_kwargs.get("gate_shrinkage", 0.1)),
        gate_tangent_scale=float(model_kwargs.get("gate_tangent_scale", math.sqrt(2.0))),
        gate_temperature=float(model_kwargs.get("gate_temperature", 1.0)),
        gate_dropout=float(model_kwargs.get("gate_dropout", 0.0)),
        **filterbank_kwargs,
    )
