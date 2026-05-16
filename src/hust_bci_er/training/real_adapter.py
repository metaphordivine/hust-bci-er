"""Compatibility wrapper for the real EEG classifier job adapter.

The implementation lives in ``_real_adapter_impl``; this module preserves the
public import path while adding run-local model provenance postprocessing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from . import _real_adapter_impl as _base

for _name in dir(_base):
    if _name.startswith("__") and _name not in {"__doc__", "__all__"}:
        continue
    globals()[_name] = getattr(_base, _name)


def _sync_patchable_globals() -> None:
    for name in ("prepare_run_manifest_context", "build_run_manifest_payload"):
        setattr(_base, name, globals()[name])


def _route_model_info(route_config_path: Path) -> tuple[str, dict[str, Any], int]:
    route_data = yaml.safe_load(Path(route_config_path).read_text(encoding="utf-8")) or {}
    model_config = route_data.get("model") or {}
    model_name = str(model_config.get("name", "")) if isinstance(model_config, dict) else str(model_config)
    model_kwargs = (
        {key: value for key, value in model_config.items() if key != "name"}
        if isinstance(model_config, dict)
        else {}
    )
    augmentation = route_data.get("augmentation")
    if isinstance(augmentation, dict) and augmentation.get("name") in {
        "split_first_fixed_crops",
        "split_first_sliding_window",
    }:
        window_sec = float(augmentation["window_sec"])
    else:
        window_sec = float(route_data.get("input_window_sec", 10))
    n_times = int(round(window_sec * _base.SFREQ))
    return model_name, model_kwargs, n_times


def _model_filterbank_params(route_config_path: Path) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    model_name, model_kwargs, n_times = _route_model_info(route_config_path)
    if not model_name:
        return model_name, model_kwargs, None
    from hust_bci_er.models.factory import build_model

    model = build_model(
        model_name,
        n_channels=30,
        n_times=n_times,
        n_classes=2,
        **model_kwargs,
    )
    params = getattr(model, "filterbank_params", None)
    return model_name, model_kwargs, params


def _patch_model_provenance(run_dir: Path, route_config_path: Path) -> None:
    model_name, model_kwargs, filterbank_params = _model_filterbank_params(route_config_path)

    model_state_path = run_dir / "model_state.local.json"
    if model_state_path.exists():
        model_state = json.loads(model_state_path.read_text(encoding="utf-8"))
        model_state["model_kwargs"] = model_kwargs
        if filterbank_params is not None:
            model_state["filterbank_params"] = filterbank_params
        model_state_path.write_text(
            json.dumps(model_state, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["model_name"] = model_name
        manifest["model_kwargs"] = model_kwargs
        if filterbank_params is not None:
            manifest["model_filterbank_params"] = filterbank_params
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def run_real_classifier_route(*args: Any, **kwargs: Any) -> RealRunArtifacts:
    _sync_patchable_globals()
    artifacts = _base.run_real_classifier_route(*args, **kwargs)
    route_config_path = kwargs.get("route_config_path")
    if route_config_path is None:
        raise TypeError("run_real_classifier_route requires route_config_path as a keyword argument")
    _patch_model_provenance(Path(artifacts.run_dir), Path(route_config_path))
    return artifacts
