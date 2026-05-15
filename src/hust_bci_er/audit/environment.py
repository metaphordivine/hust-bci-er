"""Runtime environment snapshots for auditable experiment runs."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from importlib import metadata
from typing import Any


def package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def torch_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda": None,
        "cudnn": None,
        "cuda_available": None,
        "device_count": None,
        "devices": [],
        "driver": None,
    }
    try:
        import torch
    except ImportError:
        return info

    info.update(
        {
            "installed": True,
            "version": str(torch.__version__),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
    )
    if torch.cuda.is_available():
        info["devices"] = [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]
    info["driver"] = nvidia_driver_version()
    return info


def nvidia_driver_version() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def pip_freeze() -> list[str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def capture_environment() -> dict[str, Any]:
    """Capture the runtime versions that can affect reproducibility."""
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": {
            "numpy": package_version("numpy"),
            "PyYAML": package_version("PyYAML"),
            "pytest": package_version("pytest"),
            "torch": package_version("torch"),
        },
        "torch": torch_environment(),
        "pip_freeze": pip_freeze(),
    }
