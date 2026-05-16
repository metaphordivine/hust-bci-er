"""Model factory backed by the route registry."""

from __future__ import annotations

from torch import nn

from .backbones.cbramod import build_cbramod
from .backbones.conformer_lite import build_conformer_lite
from .backbones.deformer_lite import build_deformer_lite
from .backbones.eegnet import build_eegnet
from .backbones.fbstcnet import build_fbstcnet
from .backbones.shallow_conv_net import build_shallow_conv_net
from .backbones.srfnet import build_srfnet


BUILDERS = {
    "eegnet": build_eegnet,
    "conformer_lite": build_conformer_lite,
    "deformer_lite": build_deformer_lite,
    "srfnet": build_srfnet,
    "shallow_conv_net": build_shallow_conv_net,
    "cbramod": build_cbramod,
    "fbstcnet": build_fbstcnet,
}


def build_model(name: str, n_channels: int, n_times: int, n_classes: int = 2, **kwargs) -> nn.Module:
    if name not in BUILDERS:
        raise KeyError(f"unknown model: {name}")
    return BUILDERS[name](n_channels=n_channels, n_times=n_times, n_classes=n_classes, **kwargs)

