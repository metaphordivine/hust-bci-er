"""FFT feature layer used by SRFNet-style branches."""

from __future__ import annotations

import torch
from torch import nn


class FFTFeatureLayer(nn.Module):
    def __init__(self, mode: str = "log_power", freq_max: int | None = None, normalize: bool = False) -> None:
        super().__init__()
        if mode not in {"magnitude", "power", "log_power"}:
            raise ValueError(f"unsupported FFT mode: {mode}")
        self.mode = mode
        self.freq_max = freq_max
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        fft = torch.fft.rfft(x, dim=-1)
        if self.mode == "magnitude":
            feats = torch.abs(fft)
        elif self.mode == "power":
            feats = torch.abs(fft) ** 2
        else:
            feats = torch.log1p(torch.abs(fft) ** 2)
        if self.freq_max is not None:
            feats = feats[..., : self.freq_max]
        if self.normalize:
            mean = feats.mean(dim=-1, keepdim=True)
            std = feats.std(dim=-1, keepdim=True).clamp_min(1e-6)
            feats = (feats - mean) / std
        return feats.unsqueeze(1)

    def compute_freq_bins(self, n_times: int) -> int:
        bins = n_times // 2 + 1
        return min(bins, self.freq_max) if self.freq_max is not None else bins

