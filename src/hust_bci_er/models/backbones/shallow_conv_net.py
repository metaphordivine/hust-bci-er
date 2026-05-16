"""ShallowConvNet / ShallowFBCSPNet for EEG decoding.

Models the FBCSP pipeline: temporal filter -> spatial filter -> square ->
mean pooling -> log -> dropout -> classifier.
"""

from __future__ import annotations

import torch
from torch import nn


class Square(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class SafeLog(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=self.eps))


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        n_filters_spat: int = 40,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        final_conv_length: int | str = "auto",
        batch_norm: bool = True,
        batch_norm_alpha: float = 0.1,
        drop_prob: float = 0.5,
    ) -> None:
        super().__init__()

        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times

        self.conv_time = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters_time,
            kernel_size=(1, filter_time_length),
            bias=True,
        )

        self.conv_spat = nn.Conv2d(
            in_channels=n_filters_time,
            out_channels=n_filters_spat,
            kernel_size=(n_chans, 1),
            bias=not batch_norm,
        )

        self.bn = (
            nn.BatchNorm2d(n_filters_spat, momentum=batch_norm_alpha, affine=True)
            if batch_norm
            else nn.Identity()
        )

        self.square = Square()
        self.pool = nn.AvgPool2d(
            kernel_size=(1, pool_time_length),
            stride=(1, pool_time_stride),
        )
        self.safe_log = SafeLog()
        self.drop = nn.Dropout(p=drop_prob)

        if final_conv_length == "auto":
            with torch.no_grad():
                dummy = torch.zeros(1, n_chans, n_times)
                feat = self._forward_features(dummy)
                final_conv_length = feat.shape[-1]

        self.classifier = nn.Conv2d(
            in_channels=n_filters_spat,
            out_channels=n_outputs,
            kernel_size=(1, final_conv_length),
            bias=True,
        )

        self._init_weights()

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = x.unsqueeze(1)          # [B, 1, C, T]
        x = self.conv_time(x)       # [B, Ft, C, T1]
        x = self.conv_spat(x)       # [B, Fs, 1, T1]
        x = self.bn(x)
        x = self.square(x)
        x = self.pool(x)
        x = self.safe_log(x)
        x = self.drop(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B, C, T] or [B, 1, C, T]
        if x.dim() == 4:
            x = x.squeeze(1)
        x = self._forward_features(x)
        x = self.classifier(x)      # [B, K, 1, 1]
        return x.squeeze(-1).squeeze(-1)

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.conv_time.weight, gain=1.0)
        nn.init.constant_(self.conv_time.bias, 0.0)
        nn.init.xavier_uniform_(self.conv_spat.weight, gain=1.0)
        if self.conv_spat.bias is not None:
            nn.init.constant_(self.conv_spat.bias, 0.0)
        if isinstance(self.bn, nn.BatchNorm2d):
            nn.init.constant_(self.bn.weight, 1.0)
            nn.init.constant_(self.bn.bias, 0.0)
        nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)
        nn.init.constant_(self.classifier.bias, 0.0)


def build_shallow_conv_net(
    n_channels: int,
    n_times: int,
    n_classes: int = 2,
    **kwargs: object,
) -> ShallowConvNet:
    return ShallowConvNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        n_filters_time=int(kwargs.get("n_filters_time", 40)),
        filter_time_length=int(kwargs.get("filter_time_length", 25)),
        n_filters_spat=int(kwargs.get("n_filters_spat", 40)),
        pool_time_length=int(kwargs.get("pool_time_length", 75)),
        pool_time_stride=int(kwargs.get("pool_time_stride", 15)),
        final_conv_length=kwargs.get("final_conv_length", "auto"),
        batch_norm=bool(kwargs.get("batch_norm", True)),
        batch_norm_alpha=float(kwargs.get("batch_norm_alpha", 0.1)),
        drop_prob=float(kwargs.get("drop_prob", 0.5)),
    )
