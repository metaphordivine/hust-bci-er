from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.config.registry import GRAPH_MODELS, TORCH_BACKBONES  # noqa: E402
from hust_bci_er.models.factory import build_model  # noqa: E402


HUST_MONTAGE_MODELS = {"dgcnn", "lggnet", "tsception"}
LONG_BACKWARD_MODELS = {"dgcnn", "fbcnet", "lggnet", "riemannian_tangent", "tsception"}


def smoke_kwargs(name: str) -> dict[str, str]:
    if name in HUST_MONTAGE_MODELS:
        return {"channel_montage": "hust_30_a2"}
    return {}


def main() -> int:
    import torch

    torch.manual_seed(0)
    x = torch.randn(2, 30, 256)
    for name in sorted(TORCH_BACKBONES | GRAPH_MODELS):
        model = build_model(name, n_channels=30, n_times=256, n_classes=2, **smoke_kwargs(name))
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
    labels = torch.tensor([0, 1])
    x_long = torch.randn(2, 30, 2500)
    criterion = torch.nn.CrossEntropyLoss()
    for name in sorted(LONG_BACKWARD_MODELS):
        model = build_model(name, n_channels=30, n_times=2500, n_classes=2, **smoke_kwargs(name))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(x_long)
        logits = out["logits"] if isinstance(out, dict) else out
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            print(f"{name}: non-finite long-window loss", file=sys.stderr)
            return 1
        loss.backward()
        optimizer.step()
        print(f"PASS {name} long_backward")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
