"""Sample-level gated fusion of SRFNet and FBSTCNet logits."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .fbstcnet import FBSTCNet
from .srfnet import SRFNet


class SRFFBSTCNetGate(nn.Module):
    """Fuse SRFNet and FBSTCNet-M predictions with a learned sample gate.

    The gate sees only model logits, not subject or trial metadata. Fusion is
    performed in probability space to keep the two component classifiers as
    peers instead of letting a raw logit scale dominate.
    """

    def __init__(
        self,
        *,
        n_channels: int = 30,
        n_times: int = 2500,
        n_classes: int = 2,
        srf: dict[str, Any] | None = None,
        fbstcnet: dict[str, Any] | None = None,
        gate_hidden_dim: int = 16,
        gate_dropout: float = 0.15,
        fusion_temperature: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if n_classes != 2:
            raise ValueError("SRFFBSTCNetGate currently expects binary classification")
        if gate_hidden_dim <= 0:
            raise ValueError("gate_hidden_dim must be positive")
        if not 0.0 <= gate_dropout < 1.0:
            raise ValueError("gate_dropout must be in [0, 1)")
        if fusion_temperature <= 0:
            raise ValueError("fusion_temperature must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        srf_kwargs = dict(srf or {})
        fbstcnet_kwargs = dict(fbstcnet or {})
        fbstcnet_kwargs.setdefault("sfreq", 250.0)
        fbstcnet_kwargs.setdefault("variant", "M")

        self.srf = SRFNet(n_channels=n_channels, n_times=n_times, **srf_kwargs)
        self.fbstcnet = FBSTCNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times,
            **fbstcnet_kwargs,
        )
        self.fusion_temperature = float(fusion_temperature)
        self.eps = float(eps)
        self.gate = nn.Sequential(
            nn.LayerNorm(n_classes * 2),
            nn.Linear(n_classes * 2, int(gate_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(gate_dropout)),
            nn.Linear(int(gate_hidden_dim), 2),
        )

    @staticmethod
    def _srf_logits(output: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(output, dict):
            logits = output.get("logits")
            if logits is None:
                raise ValueError("SRFNet output dictionary does not contain logits")
            return logits
        return output

    def component_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        srf_logits = self._srf_logits(self.srf(x))
        fbstcnet_logits = self.fbstcnet(x)
        return srf_logits, fbstcnet_logits

    def component_weights(self, srf_logits: torch.Tensor, fbstcnet_logits: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([srf_logits.detach(), fbstcnet_logits.detach()], dim=-1)
        gate_logits = self.gate(gate_input) / self.fusion_temperature
        return torch.softmax(gate_logits, dim=-1)

    def forward(self, x: torch.Tensor, return_components: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        srf_logits, fbstcnet_logits = self.component_logits(x)
        weights = self.component_weights(srf_logits, fbstcnet_logits)
        probs = torch.stack(
            [
                torch.softmax(srf_logits, dim=-1),
                torch.softmax(fbstcnet_logits, dim=-1),
            ],
            dim=1,
        )
        fused = (weights.unsqueeze(-1) * probs).sum(dim=1)
        logits = torch.log(fused.clamp_min(self.eps))
        if return_components:
            return {
                "logits": logits,
                "srf_logits": srf_logits,
                "fbstcnet_logits": fbstcnet_logits,
                "component_weights": weights,
            }
        return logits


def build_srf_fbstcnet_gate(
    n_channels: int, n_times: int, n_classes: int = 2, **kwargs: object
) -> SRFFBSTCNetGate:
    return SRFFBSTCNetGate(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        srf=kwargs.get("srf") if isinstance(kwargs.get("srf"), dict) else None,
        fbstcnet=kwargs.get("fbstcnet") if isinstance(kwargs.get("fbstcnet"), dict) else None,
        gate_hidden_dim=int(kwargs.get("gate_hidden_dim", 16)),
        gate_dropout=float(kwargs.get("gate_dropout", 0.15)),
        fusion_temperature=float(kwargs.get("fusion_temperature", 1.0)),
        eps=float(kwargs.get("eps", 1e-6)),
    )
