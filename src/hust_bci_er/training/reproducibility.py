"""Deterministic runtime controls for training and evaluation."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


UINT32_MOD = 2**32


@dataclass(frozen=True)
class ReproducibilityConfig:
    seed: int
    deterministic_algorithms: bool = True
    torch_warn_only: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    dataloader_worker_seed_base: int | None = None
    batch_order: str = "seeded_sampler_or_shuffle_false"

    def normalized_worker_seed_base(self) -> int:
        return int(self.seed if self.dataloader_worker_seed_base is None else self.dataloader_worker_seed_base)


def reproducibility_manifest(config: ReproducibilityConfig) -> dict[str, Any]:
    data = asdict(config)
    worker_base = config.normalized_worker_seed_base()
    data.update(
        {
            "python_seed": int(config.seed),
            "numpy_seed": int(config.seed),
            "torch_seed": int(config.seed),
            "dataloader_worker_seed_base": worker_base,
            "batch_order": {
                "policy": config.batch_order,
                "sampler_seed": int(config.seed),
            },
        }
    )
    return data


def dataloader_worker_seed(base_seed: int, worker_id: int) -> int:
    return (int(base_seed) + int(worker_id)) % UINT32_MOD


def seed_dataloader_worker(worker_id: int, *, base_seed: int) -> None:
    seed = dataloader_worker_seed(base_seed, worker_id)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except Exception:
        return
    torch.manual_seed(seed)


def make_torch_generator(seed: int):
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def apply_reproducibility(config: ReproducibilityConfig) -> dict[str, Any]:
    """Apply deterministic settings and return the manifest-ready settings."""
    os.environ.setdefault("PYTHONHASHSEED", str(int(config.seed)))
    random.seed(int(config.seed))
    np.random.seed(int(config.seed))

    applied = reproducibility_manifest(config)
    applied["pythonhashseed_env"] = os.environ.get("PYTHONHASHSEED")
    applied["torch_available"] = False

    try:
        import torch
    except Exception:
        return applied

    torch.manual_seed(int(config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed))
    torch.use_deterministic_algorithms(bool(config.deterministic_algorithms), warn_only=bool(config.torch_warn_only))
    torch.backends.cudnn.deterministic = bool(config.cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark)
    applied["torch_available"] = True
    return applied
