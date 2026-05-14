"""Model factory backed by the route registry."""

from __future__ import annotations

from torch import nn

from .backbones.conformer_lite import build_conformer_lite
from .backbones.deformer_lite import build_deformer_lite
from .backbones.eegnet import build_eegnet
from .backbones.srfnet import build_srfnet


BUILDERS = {
    "eegnet": build_eegnet,
    "conformer_lite": build_conformer_lite,
    "deformer_lite": build_deformer_lite,
    "srfnet": build_srfnet,
}


def build_model(name: str, n_channels: int, n_times: int, n_classes: int = 2, **kwargs) -> nn.Module:
    if name not in BUILDERS:
        raise KeyError(f"unknown model: {name}")
    return BUILDERS[name](n_channels=n_channels, n_times=n_times, n_classes=n_classes, **kwargs)

