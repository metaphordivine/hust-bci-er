"""Adaptive BatchNorm utilities.

AdaBN updates BatchNorm running statistics on target-domain inputs without
using target labels. It is an adaptation component, not a feature shortcut.
"""

from __future__ import annotations

import torch
from torch import nn


def reset_batchnorm_stats(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            child.reset_running_stats()


@torch.no_grad()
def adapt_batchnorm(model: nn.Module, batches, device: str | torch.device = "cpu") -> None:
    model.to(device)
    model.train()
    for batch in batches:
        x = batch["features"] if isinstance(batch, dict) else batch[0]
        model(x.to(device))

