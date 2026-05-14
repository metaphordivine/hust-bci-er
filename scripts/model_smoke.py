from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.config.registry import TORCH_BACKBONES  # noqa: E402
from hust_bci_er.models.factory import build_model  # noqa: E402


def main() -> int:
    import torch

    torch.manual_seed(0)
    x = torch.randn(2, 30, 256)
    for name in sorted(TORCH_BACKBONES):
        model = build_model(name, n_channels=30, n_times=256, n_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(x)
        logits = out["logits"] if isinstance(out, dict) else out
        if tuple(logits.shape) != (2, 2):
            print(f"{name}: unexpected output shape {tuple(logits.shape)}", file=sys.stderr)
            return 1
        if not torch.isfinite(logits).all():
            print(f"{name}: non-finite output", file=sys.stderr)
            return 1
        print(f"PASS {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
