"""Writers for run manifests that lock experiment artifacts."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from hust_bci_er.audit.environment import capture_environment
from hust_bci_er.audit.manifest import sha256_file
from hust_bci_er.evaluation.crop_policy import route_crop_policy_manifest
from hust_bci_er.training.reproducibility import ReproducibilityConfig, apply_reproducibility


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML document must be a mapping: {path}")
    return data


def require_route_value(data: Mapping[str, Any], *path: str) -> Any:
    current: Any = data
    traversed: list[str] = []
    for key in path:
        traversed.append(key)
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"route config missing required field: {'.'.join(traversed)}")
        current = current[key]
    return current


def require_route_mapping(data: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    value = require_route_value(data, *path)
    if not isinstance(value, Mapping):
        raise ValueError(f"route config field must be a mapping: {'.'.join(path)}")
    return value


def repo_root_from_route(route_config_path: Path) -> Path:
    current = route_config_path.resolve().parent
    markers = ("pyproject.toml", "AGENTS.md")
    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in markers) and (candidate / "configs" / "routes" / "models").is_dir():
            return candidate
    raise ValueError(f"could not find repository root for route config: {route_config_path}")


def git_commit(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True, timeout=5)
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "UNKNOWN"


def command_string(command: str | Sequence[str]) -> str:
    if isinstance(command, str):
        return command
    return " ".join(str(item) for item in command)


def relative_to_run(path: Path, run_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(run_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"run artifact must stay under run_dir: {path}") from exc


def evidence_manifest_path(path: Path, *, run_dir: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        pass
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"evidence path must stay under run_dir or repository root: {path}") from exc


def default_checkpoint_selection(route_data: Mapping[str, Any]) -> dict[str, Any]:
    training = route_data.get("training") if isinstance(route_data, Mapping) else None
    early = training.get("early_stopping") if isinstance(training, Mapping) else None
    monitor = "val_loss"
    mode = "min"
    if isinstance(early, Mapping):
        monitor = str(early.get("monitor", monitor))
        mode = str(early.get("mode", mode))
    return {
        "rule": "best_monitored_epoch",
        "monitor": monitor,
        "mode": mode,
        "tie_break": "earliest_epoch",
        "restore_best": True,
    }


def route_model_provenance(route_data: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    model = route_data.get("model")
    if isinstance(model, Mapping):
        model_name = str(model.get("name") or "")
        if model_name == "score_fusion":
            return model_name, {}
        return model_name, {str(key): value for key, value in model.items() if key != "name"}
    return str(model or ""), {}


def environment_lock_payload(root: Path) -> dict[str, Any] | None:
    files = []
    for name in ["environment.lock", "requirements.lock"]:
        path = root / name
        if path.exists():
            files.append({"path": name, "sha256": sha256_file(path)})
    if not files:
        return None
    return {"files": files}


def group_keys_payload(keys: Sequence[str] | None, *, default: Sequence[str] = ("subject_id",)) -> list[str]:
    values = list(default if keys is None else keys)
    if not values or any(not isinstance(key, str) or not key for key in values):
        raise ValueError("group keys must be a non-empty sequence of strings")
    return values


@dataclass(frozen=True)
class RunManifestContext:
    root: Path
    run_dir: Path
    route_config_path: Path
    route_data: Mapping[str, Any]
    route_id: str
    source_seed: int
    seed: int
    source_split_id: str
    split_id: str
    snapshot_path: Path
    dataset_manifest_path: Path
    split_manifest_path: Path
    prediction_csv: Path


def prepare_run_manifest_context(
    *,
    route_config_path: Path,
    run_dir: Path,
    prediction_csv: Path,
    dataset_manifest_path: Path | None = None,
    split_manifest_path: Path | None = None,
    split_id: str | None = None,
    source_split_id: str | None = None,
    seed: int | None = None,
    source_seed: int | None = None,
) -> RunManifestContext:
    route_config_path = route_config_path.resolve()
    root = repo_root_from_route(route_config_path)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    route_data = load_yaml_mapping(route_config_path)
    route_id = str(require_route_value(route_data, "route_id"))
    source_seed_value = int(require_route_value(route_data, "seed") if source_seed is None else source_seed)
    seed_value = int(source_seed_value if seed is None else seed)
    source_split_id_value = source_split_id or str(require_route_value(route_data, "split_id"))
    split_id_value = split_id or source_split_id_value

    snapshot_path = run_dir / "config_snapshot.yaml"
    shutil.copyfile(route_config_path, snapshot_path)

    dataset_version = str(require_route_value(route_data, "dataset_version"))
    dataset_path = dataset_manifest_path or root / "configs" / "datasets" / f"{dataset_version}.yaml"
    split_path = split_manifest_path
    if split_path is None:
        if split_id_value != source_split_id_value:
            raise ValueError("split_manifest_path is required when split_id overrides the route split_id")
        split_path = root / "configs" / "splits" / f"{source_split_id_value}.yaml"

    prediction_path = prediction_csv.resolve()
    if not prediction_path.exists():
        raise FileNotFoundError(f"prediction_csv not found: {prediction_path}")

    return RunManifestContext(
        root=root,
        run_dir=run_dir,
        route_config_path=route_config_path,
        route_data=route_data,
        route_id=route_id,
        source_seed=source_seed_value,
        seed=seed_value,
        source_split_id=source_split_id_value,
        split_id=split_id_value,
        snapshot_path=snapshot_path,
        dataset_manifest_path=dataset_path,
        split_manifest_path=split_path,
        prediction_csv=prediction_path,
    )


def build_run_manifest_payload(
    context: RunManifestContext,
    *,
    metrics: Mapping[str, float],
    command: str | Sequence[str],
    score_matrix_csv: Path | None = None,
    environment: Mapping[str, Any] | None = None,
    determinism: Mapping[str, Any] | None = None,
    checkpoint_selection: Mapping[str, Any] | None = None,
    crop_policy: Mapping[str, Any] | None = None,
    top4_group_keys: Sequence[str] | None = None,
    metric_group_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    require_route_mapping(context.route_data, "evaluation")
    primary_metric = str(require_route_value(context.route_data, "evaluation", "primary_metric"))
    model_name, model_kwargs = route_model_provenance(context.route_data)
    manifest: dict[str, Any] = {
        "audit_schema_version": 2,
        "route_id": context.route_id,
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "git_commit": git_commit(context.root),
        "config_path": context.route_config_path.relative_to(context.root).as_posix(),
        "config_snapshot_path": context.snapshot_path.relative_to(context.run_dir).as_posix(),
        "config_sha256": sha256_file(context.snapshot_path),
        "dataset_manifest_path": evidence_manifest_path(context.dataset_manifest_path, run_dir=context.run_dir, root=context.root),
        "dataset_manifest_sha256": sha256_file(context.dataset_manifest_path),
        "split_manifest_path": evidence_manifest_path(context.split_manifest_path, run_dir=context.run_dir, root=context.root),
        "split_id": context.split_id,
        "source_split_id": context.source_split_id,
        "split_sha256": sha256_file(context.split_manifest_path),
        "seed": context.seed,
        "source_seed": context.source_seed,
        "command": command_string(command),
        "primary_metric": primary_metric,
        "metrics": dict(metrics),
        "prediction_csv": relative_to_run(context.prediction_csv, context.run_dir),
        "prediction_sha256": sha256_file(context.prediction_csv),
        "prediction_record_level": "trial",
        "top4_group_keys": group_keys_payload(top4_group_keys),
        "environment": dict(environment or capture_environment()),
        "determinism": dict(determinism or apply_reproducibility(ReproducibilityConfig(seed=context.seed))),
        "checkpoint_selection": dict(checkpoint_selection or default_checkpoint_selection(context.route_data)),
        "crop_policy": dict(crop_policy or route_crop_policy_manifest(context.route_data, seed=context.seed)),
    }
    if metric_group_keys is not None:
        manifest["metric_group_keys"] = group_keys_payload(metric_group_keys)
    elif top4_group_keys is not None:
        manifest["metric_group_keys"] = manifest["top4_group_keys"]
    lock = environment_lock_payload(context.root)
    if lock is not None:
        manifest["environment_lock"] = lock
    if score_matrix_csv is not None:
        score_matrix_path = score_matrix_csv.resolve()
        if not score_matrix_path.exists():
            raise FileNotFoundError(f"score_matrix_csv not found: {score_matrix_path}")
        manifest["metric_inputs"] = {
            "score_matrix_csv": relative_to_run(score_matrix_path, context.run_dir),
            "score_matrix_sha256": sha256_file(score_matrix_path),
        }
    return manifest


def write_run_manifest(
    *,
    route_config_path: Path,
    run_dir: Path,
    prediction_csv: Path,
    metrics: Mapping[str, float],
    command: str | Sequence[str],
    dataset_manifest_path: Path | None = None,
    split_manifest_path: Path | None = None,
    split_id: str | None = None,
    source_split_id: str | None = None,
    seed: int | None = None,
    source_seed: int | None = None,
    score_matrix_csv: Path | None = None,
    environment: Mapping[str, Any] | None = None,
    determinism: Mapping[str, Any] | None = None,
    checkpoint_selection: Mapping[str, Any] | None = None,
    crop_policy: Mapping[str, Any] | None = None,
    top4_group_keys: Sequence[str] | None = None,
    metric_group_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write ``manifest.json`` for a finished run and return its payload."""
    context = prepare_run_manifest_context(
        route_config_path=route_config_path,
        run_dir=run_dir,
        prediction_csv=prediction_csv,
        dataset_manifest_path=dataset_manifest_path,
        split_manifest_path=split_manifest_path,
        split_id=split_id,
        source_split_id=source_split_id,
        seed=seed,
        source_seed=source_seed,
    )
    manifest = build_run_manifest_payload(
        context,
        metrics=metrics,
        command=command,
        score_matrix_csv=score_matrix_csv,
        environment=environment,
        determinism=determinism,
        checkpoint_selection=checkpoint_selection,
        crop_policy=crop_policy,
        top4_group_keys=top4_group_keys,
        metric_group_keys=metric_group_keys,
    )
    (context.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
