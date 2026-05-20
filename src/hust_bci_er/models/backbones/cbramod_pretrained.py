"""Official CBraMod pretrained checkpoint adapter.

This module keeps the public route separate from the local ``cbramod`` scratch
backbone.  It mirrors the upstream CBraMod encoder state-dict names so the
author-released ``pretrained_weights.pth`` can be loaded with strict key
matching, then adds a repository-native classifier head for HUST routes.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


OFFICIAL_CBRAMOD_REPO = "https://github.com/wjq-learning/CBraMod"
OFFICIAL_CBRAMOD_WEIGHT_URL = "https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth"
OFFICIAL_CBRAMOD_WEIGHT_SHA256 = "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_torch_state_dict(path: Path) -> dict[str, Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"CBraMod pretrained checkpoint must be a state-dict mapping: {path}")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in payload.items()):
        raise ValueError(f"CBraMod pretrained checkpoint contains non-tensor state entries: {path}")
    return dict(payload)


def _get_clones(module: nn.Module, n_layers: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_layers)])


class OfficialTransformerEncoder(nn.Module):
    """Transformer encoder with the upstream CBraMod state-dict layout."""

    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module | None = None) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Tensor | None = None) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class OfficialTransformerEncoderLayer(nn.Module):
    """Criss-cross transformer layer compatible with the official checkpoint."""

    def __init__(
        self,
        d_model: int = 200,
        nhead: int = 8,
        dim_feedforward: int = 800,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("CBraMod d_model must be even for criss-cross attention")
        if nhead % 2 != 0:
            raise ValueError("CBraMod nhead must be even for criss-cross attention")
        self.self_attn_s = nn.MultiheadAttention(d_model // 2, nhead // 2, dropout=dropout, bias=bias, batch_first=batch_first)
        self.self_attn_t = nn.MultiheadAttention(d_model // 2, nhead // 2, dropout=dropout, bias=bias, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, x: Tensor, attn_mask: Tensor | None) -> Tensor:
        batch, channels, patches, width = x.shape
        xs = x[:, :, :, : width // 2]
        xt = x[:, :, :, width // 2 :]
        xs = xs.transpose(1, 2).contiguous().view(batch * patches, channels, width // 2)
        xt = xt.contiguous().view(batch * channels, patches, width // 2)
        xs = self.self_attn_s(xs, xs, xs, attn_mask=attn_mask, need_weights=False)[0]
        xt = self.self_attn_t(xt, xt, xt, attn_mask=attn_mask, need_weights=False)[0]
        xs = xs.contiguous().view(batch, patches, channels, width // 2).transpose(1, 2)
        xt = xt.contiguous().view(batch, channels, patches, width // 2)
        return self.dropout1(torch.cat((xs, xt), dim=3))

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))


class OfficialPatchEmbedding(nn.Module):
    """Patch embedding module compatible with the official checkpoint."""

    def __init__(self, in_dim: int = 200, out_dim: int = 200, d_model: int = 200, seq_len: int = 30) -> None:
        super().__init__()
        del out_dim, seq_len
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3), groups=d_model),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(in_dim // 2 + 1, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, channels, patches, patch_size = x.shape
        mask_x = x if mask is None else x.clone()
        if mask is not None:
            mask_x[mask == 1] = self.mask_encoding.to(device=x.device, dtype=x.dtype)

        conv_x = mask_x.contiguous().view(batch, 1, channels * patches, patch_size)
        patch_emb = self.proj_in(conv_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(batch, channels, patches, self.d_model)

        spectral_x = conv_x.contiguous().view(batch * channels * patches, patch_size)
        spectral = torch.fft.rfft(spectral_x, dim=-1, norm="forward").abs()
        spectral = spectral.contiguous().view(batch, channels, patches, patch_size // 2 + 1)
        patch_emb = patch_emb + self.spectral_proj(spectral)

        pos = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        return patch_emb + pos.permute(0, 2, 3, 1)


class OfficialCBraModEncoder(nn.Module):
    """Official CBraMod encoder shape and state-dict names."""

    def __init__(
        self,
        in_dim: int = 200,
        out_dim: int = 200,
        d_model: int = 200,
        dim_feedforward: int = 800,
        seq_len: int = 30,
        n_layer: int = 12,
        nhead: int = 8,
    ) -> None:
        super().__init__()
        self.patch_embedding = OfficialPatchEmbedding(in_dim, out_dim, d_model, seq_len)
        layer = OfficialTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
            activation=F.gelu,
        )
        self.encoder = OfficialTransformerEncoder(layer, num_layers=n_layer)
        self.proj_out = nn.Sequential(nn.Linear(d_model, out_dim))
        self.apply(_weights_init)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        patch_emb = self.patch_embedding(x, mask)
        return self.proj_out(self.encoder(patch_emb))


class CBraModPretrainedClassifier(nn.Module):
    """Official CBraMod pretrained encoder plus a HUST classifier head."""

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        patch_size: int = 200,
        out_dim: int = 200,
        d_model: int = 200,
        dim_feedforward: int = 800,
        n_layer: int = 12,
        nhead: int = 8,
        classifier: str = "all_patch_reps",
        classifier_hidden_dim: int = 800,
        classifier_mid_dim: int = 200,
        dropout: float = 0.1,
        load_pretrained: bool = False,
        pretrained_checkpoint_path: str | None = None,
        pretrained_sha256: str | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        if patch_size != 200:
            raise ValueError("official CBraMod pretrained weights require patch_size=200")
        if classifier not in {"all_patch_reps", "avgpooling_patch_reps"}:
            raise ValueError("classifier must be all_patch_reps or avgpooling_patch_reps")
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.patch_size = patch_size
        self.n_patch = (n_times + patch_size - 1) // patch_size
        self.n_times_padded = self.n_patch * patch_size
        self.classifier_name = classifier
        self.external_weight_provenance: dict[str, Any] | None = None

        self.encoder = OfficialCBraModEncoder(
            in_dim=patch_size,
            out_dim=out_dim,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            seq_len=n_chans,
            n_layer=n_layer,
            nhead=nhead,
        )
        if load_pretrained:
            self._load_pretrained_checkpoint(pretrained_checkpoint_path, pretrained_sha256)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if classifier == "avgpooling_patch_reps":
            self.classifier = nn.Sequential(
                nn.LayerNorm(out_dim),
                nn.Linear(out_dim, n_outputs),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_chans * self.n_patch * out_dim, classifier_hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, classifier_mid_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_mid_dim, n_outputs),
            )

    def _load_pretrained_checkpoint(self, checkpoint_path: str | None, expected_sha256: str | None) -> None:
        if not checkpoint_path:
            raise ValueError("pretrained_checkpoint_path is required when load_pretrained=true")
        if not expected_sha256:
            raise ValueError("pretrained_sha256 is required when load_pretrained=true")
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"CBraMod pretrained checkpoint not found: {path}")
        actual_sha = _sha256_file(path)
        if actual_sha.lower() != str(expected_sha256).lower():
            raise ValueError(
                "CBraMod pretrained checkpoint sha256 mismatch: "
                f"expected {expected_sha256.lower()}, got {actual_sha.lower()}"
            )
        state_dict = _load_torch_state_dict(path)
        self.encoder.load_state_dict(state_dict, strict=True)
        self.external_weight_provenance = {
            "model": "CBraMod",
            "source_repo": OFFICIAL_CBRAMOD_REPO,
            "source_weight_url": OFFICIAL_CBRAMOD_WEIGHT_URL,
            "checkpoint_path": str(path),
            "checkpoint_sha256": actual_sha,
            "checkpoint_size_bytes": path.stat().st_size,
            "load_policy": "strict_encoder_state_dict",
            "tokenizer_provenance": "not_applicable_patch_embedding_in_checkpoint",
            "pretraining_scope": "author_released_foundation_checkpoint",
        }

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        batch, channels, time = x.shape
        if channels != self.n_chans:
            raise ValueError(f"CBraMod pretrained expected {self.n_chans} channels, got {channels}")
        if time > self.n_times_padded:
            raise ValueError(f"CBraMod pretrained expected <= {self.n_times_padded} samples, got {time}")
        if time < self.n_times_padded:
            x = F.pad(x, (0, self.n_times_padded - time))
        x = x.view(batch, channels, self.n_patch, self.patch_size)
        feats = self.encoder(x, mask=mask)
        if self.classifier_name == "avgpooling_patch_reps":
            feats = feats.mean(dim=(1, 2))
        return self.classifier(feats)


def _weights_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def build_cbramod_pretrained(
    n_channels: int,
    n_times: int,
    n_classes: int = 2,
    **kwargs: object,
) -> CBraModPretrainedClassifier:
    return CBraModPretrainedClassifier(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        patch_size=int(kwargs.get("patch_size", 200)),
        out_dim=int(kwargs.get("out_dim", 200)),
        d_model=int(kwargs.get("d_model", 200)),
        dim_feedforward=int(kwargs.get("dim_feedforward", 800)),
        n_layer=int(kwargs.get("n_layer", 12)),
        nhead=int(kwargs.get("nhead", 8)),
        classifier=str(kwargs.get("classifier", "all_patch_reps")),
        classifier_hidden_dim=int(kwargs.get("classifier_hidden_dim", 800)),
        classifier_mid_dim=int(kwargs.get("classifier_mid_dim", 200)),
        dropout=float(kwargs.get("dropout", 0.1)),
        load_pretrained=bool(kwargs.get("load_pretrained", False)),
        pretrained_checkpoint_path=(
            str(kwargs["pretrained_checkpoint_path"])
            if kwargs.get("pretrained_checkpoint_path") is not None
            else None
        ),
        pretrained_sha256=str(kwargs["pretrained_sha256"]) if kwargs.get("pretrained_sha256") is not None else None,
        freeze_encoder=bool(kwargs.get("freeze_encoder", False)),
    )
