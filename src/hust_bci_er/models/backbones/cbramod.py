"""CBraMod-style scratch model for EEG decoding.

This module implements a patch embedding plus criss-cross transformer backbone
inspired by CBraMod. It does not load the original pretrained foundation-model
checkpoint, tokenizer artifacts, or external-weight provenance.
"""

from __future__ import annotations

import torch
from torch import nn


def _conv1d_output_length(length: int, *, kernel_size: int, stride: int, padding: int) -> int:
    return ((length + 2 * padding - kernel_size) // stride) + 1


class PatchEmbeddingCBraMod(nn.Module):
    """Time-domain + frequency-domain patch embedding with ACPE."""

    def __init__(
        self,
        patch_size: int = 200,
        drop_prob: float = 0.1,
        conv_out_channels: int = 25,
        conv_kernels: tuple[tuple[int, int, int, int], ...] = (
            (49, 25, 0, 24),
            (3, 1, 0, 1),
            (3, 1, 0, 1),
        ),
        d_model: int = 200,
        group_norm_groups: int = 5,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        if conv_out_channels <= 0:
            raise ValueError(f"CBraMod requires positive conv_out_channels, got {conv_out_channels}")
        if group_norm_groups <= 0:
            raise ValueError(f"CBraMod requires positive group_norm_groups, got {group_norm_groups}")
        if conv_out_channels % group_norm_groups != 0:
            raise ValueError(
                "CBraMod requires conv_out_channels divisible by group_norm_groups, "
                f"got conv_out_channels={conv_out_channels} and group_norm_groups={group_norm_groups}"
            )

        time_branch_width = patch_size
        for kernel_size, stride, _, padding in conv_kernels:
            time_branch_width = _conv1d_output_length(
                time_branch_width,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        time_branch_dim = conv_out_channels * time_branch_width
        if time_branch_dim != d_model:
            raise ValueError(
                "CBraMod patch embedding requires conv_out_channels * temporal_width "
                f"to equal d_model, got {time_branch_dim} vs {d_model}"
            )

        # Time-domain conv branch: [B, 1, C*N, P] -> [B, 25, C*N, P']
        layers: list[nn.Module] = []
        in_ch = 1
        for kernel_size, stride, _, padding in conv_kernels:
            layers.append(
                nn.Conv2d(
                    in_ch, conv_out_channels,
                    kernel_size=(1, kernel_size),
                    stride=(1, stride),
                    padding=(0, padding),
                )
            )
            layers.append(nn.GroupNorm(group_norm_groups, conv_out_channels))
            layers.append(nn.GELU())
            in_ch = conv_out_channels
        self.proj_in = nn.Sequential(*layers)

        # Frequency-domain branch
        self.spectral_proj = nn.Sequential(
            nn.Linear(patch_size // 2 + 1, d_model),
            nn.Dropout(drop_prob),
        )

        # ACPE: depthwise 2D conv positional encoding
        self.positional_encoding = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=(19, 7),
            stride=(1, 1),
            padding=(9, 3),
            groups=d_model,
        )

        self.mask_encoding = nn.Parameter(
            torch.zeros(patch_size), requires_grad=False
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, C, N, P]
        B, C, N, P = x.shape

        if mask is not None:
            x = x.clone()
            x[mask == 1] = self.mask_encoding.to(device=x.device, dtype=x.dtype)

        # Time-domain branch
        x_conv = x.reshape(B, 1, C * N, P)
        patch_emb = self.proj_in(x_conv)          # [B, 25, C*N, P']
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous()
        patch_emb = patch_emb.reshape(B, C, N, -1)  # [B, C, N, d_model]

        # Frequency-domain branch
        spectral = torch.fft.rfft(x, dim=-1, norm="forward")
        spectral = torch.abs(spectral)
        spectral_emb = self.spectral_proj(spectral)  # [B, C, N, d_model]

        # Fusion
        patch_emb = patch_emb + spectral_emb

        # ACPE
        pos_in = patch_emb.permute(0, 3, 1, 2)   # [B, D, C, N]
        pos = self.positional_encoding(pos_in)    # [B, D, C, N]
        pos = pos.permute(0, 2, 3, 1)            # [B, C, N, D]

        return patch_emb + pos


class CrissCrossTransformerLayer(nn.Module):
    """One criss-cross transformer layer with spatial + temporal attention."""

    def __init__(
        self,
        d_model: int = 200,
        nhead: int = 8,
        dim_feedforward: int = 800,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % 2 == 0
        assert nhead % 2 == 0

        self.d_half = d_model // 2
        self.half_heads = nhead // 2

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.s_attn = nn.MultiheadAttention(
            embed_dim=self.d_half,
            num_heads=self.half_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.t_attn = nn.MultiheadAttention(
            embed_dim=self.d_half,
            num_heads=self.half_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N, D]
        B, C, N, D = x.shape

        residual = x
        z = self.norm1(x)

        z_s, z_t = torch.split(z, self.d_half, dim=-1)

        # Spatial attention: over channels for each time patch
        # [B, C, N, D/2] -> [B*N, C, D/2]
        s = z_s.permute(0, 2, 1, 3).reshape(B * N, C, self.d_half)
        s_out, _ = self.s_attn(s, s, s, need_weights=False)
        s_out = s_out.reshape(B, N, C, self.d_half).permute(0, 2, 1, 3)

        # Temporal attention: over time patches for each channel
        # [B, C, N, D/2] -> [B*C, N, D/2]
        t = z_t.reshape(B * C, N, self.d_half)
        t_out, _ = self.t_attn(t, t, t, need_weights=False)
        t_out = t_out.reshape(B, C, N, self.d_half)

        attn_out = torch.cat([s_out, t_out], dim=-1)
        x = residual + self.dropout1(attn_out)

        residual = x
        z = self.norm2(x)
        x = residual + self.dropout2(self.ffn(z))

        return x


class CBraMod(nn.Module):
    """CBraMod-style scratch backbone with criss-cross attention."""

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        patch_size: int = 200,
        d_model: int = 200,
        emb_dim: int = 200,
        n_layer: int = 12,
        nhead: int = 8,
        dim_feedforward: int = 800,
        drop_prob: float = 0.1,
        classifier_pooling: str = "flatten",
        conv_out_channels: int = 25,
        group_norm_groups: int = 5,
    ) -> None:
        super().__init__()
        if classifier_pooling not in {"flatten", "mean"}:
            raise ValueError(f"classifier_pooling must be flatten or mean, got {classifier_pooling}")
        self.classifier_pooling = classifier_pooling

        # Pad to a multiple of patch_size when n_times is not exact
        n_patch = (n_times + patch_size - 1) // patch_size
        self.n_times_raw = n_times
        self.n_times_padded = n_patch * patch_size

        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.patch_size = patch_size
        self.n_patch = n_patch

        self.patch_embedding = PatchEmbeddingCBraMod(
            patch_size=patch_size,
            drop_prob=drop_prob,
            d_model=d_model,
            conv_out_channels=conv_out_channels,
            group_norm_groups=group_norm_groups,
        )

        self.encoder = nn.ModuleList([
            CrissCrossTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=drop_prob,
            )
            for _ in range(n_layer)
        ])

        self.proj_out = nn.Linear(d_model, emb_dim)

        if classifier_pooling == "mean":
            self.classifier = nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, n_outputs),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_chans * self.n_patch * emb_dim, n_outputs),
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Accept [B, C, T] or [B, 1, C, T]
        if x.dim() == 4:
            x = x.squeeze(1)

        B, C, T = x.shape

        if T > self.n_times_padded:
            raise ValueError(
                f"CBraMod expected <= {self.n_times_padded} samples after padding, got {T}"
            )

        # Pad time dimension to multiple of patch_size
        if T < self.n_times_padded:
            x = torch.nn.functional.pad(x, (0, self.n_times_padded - T))

        x = x.view(B, C, self.n_patch, self.patch_size)  # [B, C, N, P]
        x = self.patch_embedding(x, mask=mask)            # [B, C, N, D]

        for layer in self.encoder:
            x = layer(x)                                  # [B, C, N, D]

        x = self.proj_out(x)                              # [B, C, N, emb_dim]
        if self.classifier_pooling == "mean":
            x = x.mean(dim=(1, 2))                         # [B, emb_dim]
        return self.classifier(x)                         # [B, K]


def build_cbramod(
    n_channels: int,
    n_times: int,
    n_classes: int = 2,
    **kwargs: object,
) -> CBraMod:
    return CBraMod(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        patch_size=int(kwargs.get("patch_size", 200)),
        d_model=int(kwargs.get("d_model", 200)),
        emb_dim=int(kwargs.get("emb_dim", 200)),
        n_layer=int(kwargs.get("n_layer", 12)),
        nhead=int(kwargs.get("nhead", 8)),
        dim_feedforward=int(kwargs.get("dim_feedforward", 800)),
        drop_prob=float(kwargs.get("drop_prob", 0.1)),
        classifier_pooling=str(kwargs.get("classifier_pooling", "flatten")),
        conv_out_channels=int(kwargs.get("conv_out_channels", 25)),
        group_norm_groups=int(kwargs.get("group_norm_groups", 5)),
    )
