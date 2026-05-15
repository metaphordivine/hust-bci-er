"""FBSTCNet: Filter-Bank Spatio-Temporal Convolutional Network.

Variants: P (power), C (connectivity), M (mixed).
Uses internal FFT-based filterbank, cropped decoding with aggregation.
"""

from __future__ import annotations

import torch
from torch import nn


def aggregate_crop_logits(crop_logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Aggregate crop-level logits to trial-level logits via mean in prob space.

    crop_logits: [B, N_crop, K]
    returns: [B, K]
    """
    prob = torch.softmax(crop_logits, dim=-1)       # [B, N_crop, K]
    prob_mean = prob.mean(dim=1)                    # [B, K]
    log_prob = torch.log(prob_mean.clamp_min(eps))  # [B, K]
    return log_prob


class FFTFilterBank(nn.Module):
    """FFT-based bandpass filter bank: [B, C, T] -> [B, Nb, C, T]."""

    def __init__(
        self,
        sfreq: float = 200.0,
        bands: list[tuple[float, float]] | None = None,
    ) -> None:
        super().__init__()
        if bands is None:
            bands = [(f, f + 4) for f in range(4, 52, 4)]
        self.sfreq = sfreq
        self.bands = bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        B, C, T = x.shape
        X = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sfreq, device=x.device)

        outs: list[torch.Tensor] = []
        for low, high in self.bands:
            mask = (freqs >= low) & (freqs <= high)
            X_band = X * mask.to(dtype=X.dtype)
            x_band = torch.fft.irfft(X_band, dim=-1, n=T)
            outs.append(x_band)

        return torch.stack(outs, dim=1)  # [B, Nb, C, T]


class FBSTC_STConvBlock(nn.Module):
    """Spatio-temporal convolution block: temporal conv + depthwise spatial conv."""

    def __init__(
        self,
        n_bands: int,
        n_chans: int,
        sfreq: float,
        F1: int = 36,
        F2: int = 36,
        alpha: float = 0.125,
    ) -> None:
        super().__init__()
        kt = int(alpha * sfreq)

        self.temporal = nn.Conv2d(
            in_channels=n_bands,
            out_channels=F1,
            kernel_size=(1, kt),
            padding=(0, kt // 2),
            bias=False,
        )

        self.spatial = nn.Conv2d(
            in_channels=F1,
            out_channels=F2,
            kernel_size=(n_chans, 1),
            groups=F1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(F2)

    def forward(self, x_fb: torch.Tensor) -> torch.Tensor:
        # x_fb: [B, Nb, C, T]
        z = self.temporal(x_fb)  # [B, F1, C, T]
        z = self.spatial(z)      # [B, F2, 1, T]
        z = self.bn(z)           # [B, F2, 1, T]
        return z


class FBSTC_PowerHead(nn.Module):
    """Power branch: square -> pool -> log -> dropout -> crop classifier."""

    def __init__(
        self,
        F2: int,
        n_outputs: int,
        alpha1: int,
        alpha2: int,
        pool_size: int = 80,
        pool_stride: int = 5,
        drop_prob: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps

        self.pool = nn.AvgPool2d(
            kernel_size=(1, pool_size),
            stride=(1, pool_stride),
        )
        self.drop = nn.Dropout(p=drop_prob)

        self.classifier = nn.Conv2d(
            in_channels=F2,
            out_channels=n_outputs,
            kernel_size=(1, alpha1),
            stride=(1, alpha2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, F2, 1, T]
        z = z * z
        z = self.pool(z)
        z = torch.log(z.clamp_min(self.eps))
        z = self.drop(z)

        logits = self.classifier(z)       # [B, K, 1, N_crop]
        logits = logits.squeeze(2)        # [B, K, N_crop]
        logits = logits.transpose(1, 2)   # [B, N_crop, K]
        return logits


class FBSTC_ConnectivityHead(nn.Module):
    """Connectivity branch: sliding window -> Pearson corr -> flatten -> dense."""

    def __init__(
        self,
        F2: int,
        n_outputs: int,
        gamma: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.F2 = F2
        self.gamma = gamma
        self.stride = max(gamma // 2, 1)
        self.eps = eps

        self.classifier = nn.Linear(F2 * F2, n_outputs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, F2, 1, T]
        z = z.squeeze(2)  # [B, F2, T]

        T = z.shape[-1]
        gamma = min(self.gamma, T)
        stride = max(gamma // 2, 1)

        windows = z.unfold(
            dimension=-1, size=gamma, step=stride
        )  # [B, F2, N_crop, gamma]

        windows = windows.permute(0, 2, 1, 3).contiguous()
        # [B, N_crop, F2, gamma]

        corr = self._pearson(windows)  # [B, N_crop, F2, F2]

        feat = corr.flatten(start_dim=2)  # [B, N_crop, F2*F2]

        logits = self.classifier(feat)  # [B, N_crop, K]
        return logits

    def _pearson(self, w: torch.Tensor) -> torch.Tensor:
        w = w - w.mean(dim=-1, keepdim=True)
        numerator = torch.matmul(w, w.transpose(-1, -2))
        power = torch.sum(w * w, dim=-1, keepdim=True)
        denom = torch.sqrt(power * power.transpose(-1, -2))
        return numerator / denom.clamp_min(self.eps)


class FBSTCNet(nn.Module):
    """FBSTCNet: Filter-Bank Spatio-Temporal Convolutional Network.

    Variants:
      - "P": power features only
      - "C": connectivity features only
      - "M": both power and connectivity (independent STConv blocks)
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
    ) -> None:
        super().__init__()
        del n_times  # time dimension handled dynamically

        if variant not in ("P", "C", "M"):
            raise ValueError(f"variant must be P, C, or M, got {variant}")
        self.variant = variant
        self.sfreq = sfreq

        self.filterbank = FFTFilterBank(sfreq=sfreq, bands=bands)

        if variant in ("P", "M"):
            self.st_p = FBSTC_STConvBlock(
                n_bands=n_bands, n_chans=n_chans, sfreq=sfreq,
                F1=F1, F2=F2, alpha=st_alpha,
            )
            self.power_head = FBSTC_PowerHead(
                F2=F2, n_outputs=n_outputs,
                alpha1=alpha1, alpha2=alpha2,
                pool_size=pool_size, pool_stride=pool_stride,
                drop_prob=power_drop_prob,
            )

        if variant in ("C", "M"):
            self.st_c = FBSTC_STConvBlock(
                n_bands=n_bands, n_chans=n_chans, sfreq=sfreq,
                F1=F1, F2=F2, alpha=st_alpha,
            )
            self.conn_head = FBSTC_ConnectivityHead(
                F2=F2, n_outputs=n_outputs, gamma=gamma, eps=conn_eps,
            )

    def forward(
        self, x: torch.Tensor, return_crop_logits: bool = False
    ) -> torch.Tensor:
        # Accept [B, C, T] or [B, 1, C, T]
        if x.dim() == 4:
            x = x.squeeze(1)

        x_fb = self.filterbank(x)  # [B, Nb, C, T]

        crop_logits_list: list[torch.Tensor] = []

        if self.variant in ("P", "M"):
            z_p = self.st_p(x_fb)
            crop_logits_p = self.power_head(z_p)
            crop_logits_list.append(crop_logits_p)

        if self.variant in ("C", "M"):
            z_c = self.st_c(x_fb)
            crop_logits_c = self.conn_head(z_c)
            crop_logits_list.append(crop_logits_c)

        crop_logits = torch.cat(crop_logits_list, dim=1)  # [B, N_total, K]

        if return_crop_logits:
            return crop_logits

        return aggregate_crop_logits(crop_logits)


def build_fbstcnet(
    n_channels: int,
    n_times: int,
    n_classes: int = 2,
    **kwargs: object,
) -> FBSTCNet:
    return FBSTCNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        sfreq=float(kwargs.get("sfreq", 200.0)),
        variant=str(kwargs.get("variant", "M")),
        n_bands=int(kwargs.get("n_bands", 12)),
        bands=kwargs.get("bands", None),
        F1=int(kwargs.get("F1", 36)),
        F2=int(kwargs.get("F2", 36)),
        st_alpha=float(kwargs.get("st_alpha", 0.125)),
        alpha1=int(kwargs.get("alpha1", 25)),
        alpha2=int(kwargs.get("alpha2", 5)),
        pool_size=int(kwargs.get("pool_size", 80)),
        pool_stride=int(kwargs.get("pool_stride", 5)),
        power_drop_prob=float(kwargs.get("power_drop_prob", 0.5)),
        gamma=int(kwargs.get("gamma", 400)),
        conn_eps=float(kwargs.get("conn_eps", 1e-6)),
    )
