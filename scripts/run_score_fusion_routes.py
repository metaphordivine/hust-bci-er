"""Orchestrate score_fusion route assembly from component route predictions.

For each score_fusion route in ``configs/routes/models/``, this script:

1. Resolves each logical component ID to a base torch_classifier route via
   the component-route map (``hust_bci_er.config.component_map``).
2. Looks for existing component-score CSV files under a well-known
   ``component_scores/`` directory, or exports them from the base route's
   predictions if ``--export-missing`` is passed.
3. Calls ``assemble_score_route.py`` to produce the fused predictions CSV.
4. Optionally runs ``repo_doctor.py experiment`` audit on the result.

Usage::

    # List score_fusion routes and their component bindings (dry-run):
    python scripts/run_score_fusion_routes.py --dry-run

    # Assemble from pre-exported component scores:
    python scripts/run_score_fusion_routes.py --component-scores-dir outputs/component_scores

    # Auto-export component scores from base route run dirs, then assemble:
    python scripts/run_score_fusion_routes.py \
        --component-scores-dir outputs/component_scores \
        --base-runs-dir outputs/candidate_all_models \
        --export-missing
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from hust_bci_er.config.component_map import base_route_for_component  # noqa: E402
from hust_bci_er.audit.manifest import sha256_file  # noqa: E402
from hust_bci_er.audit.run_manifest import build_run_manifest_payload, prepare_run_manifest_context  # noqa: E402
from hust_bci_er.evaluation.report import build_metric_report, write_metric_report  # noqa: E402
from hust_bci_er.inference.clean_score_routes import score_route_by_id  # noqa: E402
from hust_bci_er.inference.score_route_assembly import assemble_score_route_rows  # noqa: E402
from hust_bci_er.inference.topk import subject_top4  # noqa: E402
from hust_bci_er.training.reproducibility import ReproducibilityConfig, reproducibility_manifest  # noqa: E402


@dataclass(frozen=True)
class SourceArtifact:
    path: Path
    kind: str
    job_id: str | None = None
    seed: int | None = None
    fold: int | None = None
    manifest_path: Path | None = None


def route_config_paths() -> list[Path]:
    return sorted((ROOT / "configs" / "routes" / "models").glob("*.yaml"))


def load_route(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"route config must be a mapping: {path}")
    return data


def route_model_name(route_data: dict[str, Any]) -> str:
    model = route_data.get("model")
    if isinstance(model, dict):
        return str(model.get("name") or "")
    return ""


def is_score_fusion_route(route_data: dict[str, Any]) -> bool:
    return route_model_name(route_data) == "score_fusion"


def score_fusion_routes() -> list[tuple[Path, dict[str, Any]]]:
    result: list[tuple[Path, dict[str, Any]]] = []
    for path in route_config_paths():
        data = load_route(path)
        if is_score_fusion_route(data):
            result.append((path, data))
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _display_path(path: Path) -> str:
    resolved_root = ROOT.resolve()
    resolved_path = path.resolve()
    if resolved_path.is_relative_to(resolved_root):
        return resolved_path.relative_to(resolved_root).as_posix()
    return path.as_posix()


def _resolve_run_artifact(manifest_path: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    run_candidate = (manifest_path.parent / value).resolve()
    if run_candidate.exists():
        return run_candidate
    repo_candidate = (ROOT / value).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return None


def _protocol_job_metadata(run_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = run_dir / "protocol_run_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        protocol_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {
        str(job.get("job_id")): dict(job)
        for job in protocol_manifest.get("jobs", [])
        if isinstance(job, dict) and job.get("job_id")
    }


def find_component_source_artifacts(run_dir: Path) -> list[SourceArtifact]:
    """Find component-producing artifacts, preferring score_matrix.csv.

    A protocol run has one artifact per materialized job under job_runs/.
    This returns all of them and injects job seed/fold metadata so later
    score-fusion alignment does not silently drop folds.
    """
    if (run_dir / "score_matrix.csv").exists():
        manifest_path = run_dir / "manifest.json"
        seed: int | None = None
        fold: int | None = None
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if type(manifest.get("seed")) is int:
                    seed = int(manifest["seed"])
                if type(manifest.get("fold")) is int:
                    fold = int(manifest["fold"])
            except json.JSONDecodeError:
                pass
        return [SourceArtifact(run_dir / "score_matrix.csv", "score_matrix", seed=seed, fold=fold, manifest_path=manifest_path if manifest_path.exists() else None)]
    if (run_dir / "predictions.csv").exists():
        manifest_path = run_dir / "manifest.json"
        return [SourceArtifact(run_dir / "predictions.csv", "predictions", manifest_path=manifest_path if manifest_path.exists() else None)]

    job_meta = _protocol_job_metadata(run_dir)
    score_matrices = sorted(run_dir.glob("job_runs/*/score_matrix.csv"))
    if score_matrices:
        sources: list[SourceArtifact] = []
        for path in score_matrices:
            job_id = path.parent.name
            job = job_meta.get(job_id, {})
            manifest_path = path.parent / "manifest.json"
            sources.append(
                SourceArtifact(
                    path,
                    "score_matrix",
                    job_id=job_id,
                    seed=int(job["seed"]) if type(job.get("seed")) is int else None,
                    fold=int(job["fold"]) if type(job.get("fold")) is int else None,
                    manifest_path=manifest_path if manifest_path.exists() else None,
                )
            )
        return sources

    predictions = sorted(run_dir.glob("job_runs/*/predictions.csv"))
    sources = []
    for path in predictions:
        job_id = path.parent.name
        job = job_meta.get(job_id, {})
        manifest_path = path.parent / "manifest.json"
        sources.append(
            SourceArtifact(
                path,
                "predictions",
                job_id=job_id,
                seed=int(job["seed"]) if type(job.get("seed")) is int else None,
                fold=int(job["fold"]) if type(job.get("fold")) is int else None,
                manifest_path=manifest_path if manifest_path.exists() else None,
            )
        )
    return sources


def _score_matrix_crop_columns(fields: set[str]) -> list[str]:
    underscored = [f"crop_{idx}" for idx in range(5)]
    if all(col in fields for col in underscored):
        return underscored
    compact = [f"crop{idx}" for idx in range(5)]
    if all(col in fields for col in compact):
        return compact
    return []


def _component_rows_from_score_matrix(source: SourceArtifact, component_id: str) -> list[dict[str, str]]:
    rows = _read_csv(source.path)
    if not rows:
        raise ValueError(f"score matrix is empty: {source.path}")
    fields = set(rows[0])
    crop_cols = _score_matrix_crop_columns(fields)
    if not {"subject_id", "trial_id", "y_true"}.issubset(fields) or len(crop_cols) != 5:
        raise ValueError(f"score matrix must include subject_id, trial_id, y_true, crop_0..crop_4: {source.path}")
    out: list[dict[str, str]] = []
    for row in rows:
        for crop_id, crop_col in enumerate(crop_cols):
            item = {
                "component_id": component_id,
                "subject_id": str(row["subject_id"]),
                "trial_id": str(row["trial_id"]),
                "crop_id": str(crop_id),
                "score": str(row[crop_col]),
                "y_true": str(row["y_true"]),
            }
            if source.seed is not None:
                item["seed"] = str(source.seed)
            elif row.get("seed") not in {None, ""}:
                item["seed"] = str(row["seed"])
            if source.fold is not None:
                item["fold"] = str(source.fold)
            elif row.get("fold") not in {None, ""}:
                item["fold"] = str(row["fold"])
            out.append(item)
    return out


def _component_rows_from_predictions(source: SourceArtifact, component_id: str) -> list[dict[str, str]]:
    rows = _read_csv(source.path)
    if not rows:
        raise ValueError(f"predictions file is empty: {source.path}")
    fields = set(rows[0])
    if not {"subject_id", "trial_id"}.issubset(fields) or not ({"score", "y_score", "probability", "logit"} & fields):
        raise ValueError(f"predictions must include subject_id, trial_id, and a score column: {source.path}")
    score_col = next(col for col in ("y_score", "score", "probability", "logit") if col in fields)
    out: list[dict[str, str]] = []
    for row in rows:
        item = {
            "component_id": component_id,
            "subject_id": str(row["subject_id"]),
            "trial_id": str(row["trial_id"]),
            "score": str(row[score_col]),
        }
        if row.get("y_true") not in {None, ""}:
            item["y_true"] = str(row["y_true"])
        if row.get("crop_id") not in {None, ""}:
            item["crop_id"] = str(row["crop_id"])
        if source.seed is not None:
            item["seed"] = str(source.seed)
        elif row.get("seed") not in {None, ""}:
            item["seed"] = str(row["seed"])
        if source.fold is not None:
            item["fold"] = str(source.fold)
        elif row.get("fold") not in {None, ""}:
            item["fold"] = str(row["fold"])
        out.append(item)
    return out


def export_component_score(
    component_id: str,
    base_route_id: str,
    source_artifacts: list[SourceArtifact],
    output_path: Path,
) -> int:
    route_config = ROOT / "configs" / "routes" / "models" / f"{base_route_id}.yaml"
    if not route_config.exists():
        print(f"  base route config not found for {component_id}: {route_config}", file=sys.stderr)
        return 1
    if not source_artifacts:
        print(f"  no source artifacts found for {component_id}", file=sys.stderr)
        return 1
    try:
        rows: list[dict[str, str]] = []
        for source in source_artifacts:
            if source.kind == "score_matrix":
                rows.extend(_component_rows_from_score_matrix(source, component_id))
            else:
                rows.extend(_component_rows_from_predictions(source, component_id))
    except Exception as exc:
        print(f"  export failed for {component_id}: {exc}", file=sys.stderr)
        return 1
    fieldnames = ["component_id"]
    for col in ("seed", "fold", "subject_id", "trial_id", "crop_id", "score", "y_true"):
        if col in {"subject_id", "trial_id", "score"} or any(row.get(col) not in {None, ""} for row in rows):
            fieldnames.append(col)
    _write_csv(output_path, rows, fieldnames)
    return 0


def assemble_score_fusion(
    route_path: Path,
    component_score_paths: dict[str, Path],
    output_dir: Path,
) -> tuple[int, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        fused_rows = assemble_score_route_rows(route_path, component_score_paths)
    except Exception as exc:
        print(f"  assembly failed: {exc}", file=sys.stderr)
        return 1, {}
    if not fused_rows:
        print("  assembly produced no rows", file=sys.stderr)
        return 1, {}

    score_matrix_evidence = _write_score_fusion_outputs(fused_rows, output_dir)
    return 0, {"score_matrix_evidence": score_matrix_evidence}


def _metadata_columns(rows: list[dict[str, str]]) -> list[str]:
    return [col for col in ("seed", "fold") if any(row.get(col) not in {None, ""} for row in rows)]


def _prediction_group_key(row: dict[str, str], metadata_cols: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(col, "")) for col in [*metadata_cols, "subject_id"])


def _trial_key(row: dict[str, str], metadata_cols: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(col, "")) for col in [*metadata_cols, "subject_id", "trial_id"])


def _write_score_fusion_outputs(fused_rows: list[dict[str, str]], output_dir: Path) -> str:
    metadata_cols = _metadata_columns(fused_rows)
    has_crop_rows = any(row.get("crop_id") not in {None, ""} for row in fused_rows)
    trial_scores: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in fused_rows:
        trial_scores.setdefault(_trial_key(row, metadata_cols), []).append(row)

    predictions: list[dict[str, str]] = []
    score_rows: list[dict[str, str]] = []
    score_matrix_evidence = "genuine" if has_crop_rows else "synthetic"

    for key, rows in sorted(trial_scores.items()):
        first = rows[0]
        scores_by_crop: dict[int, float] = {}
        if has_crop_rows:
            for row in rows:
                crop_raw = row.get("crop_id")
                if crop_raw in {None, ""}:
                    continue
                scores_by_crop[int(float(crop_raw))] = float(row["score"])
        if len(scores_by_crop) == 5 and sorted(scores_by_crop) == [0, 1, 2, 3, 4]:
            crop_scores = [scores_by_crop[idx] for idx in range(5)]
        else:
            score_matrix_evidence = "synthetic"
            mean_source = float(first["score"])
            crop_scores = [mean_source for _idx in range(5)]

        mean_score = sum(crop_scores) / len(crop_scores)
        pred = {
            "route_id": str(first["route_id"]),
            "subject_id": str(first["subject_id"]),
            "trial_id": str(first["trial_id"]),
            "y_score": f"{mean_score:.12g}",
            "y_pred": str(int(mean_score >= 0.0)),
        }
        if first.get("y_true") not in {None, ""}:
            pred["y_true"] = str(first["y_true"])
        for col in metadata_cols:
            pred[col] = str(first.get(col, ""))
        predictions.append(pred)

        score_row = {col: str(first.get(col, "")) for col in metadata_cols}
        score_row.update(
            {
                "subject_id": str(first["subject_id"]),
                "trial_id": str(first["trial_id"]),
                "y_true": str(first.get("y_true", "0")),
            }
        )
        for idx, score in enumerate(crop_scores):
            score_row[f"crop_{idx}"] = f"{score:.12g}"
            score_row[f"crop_{idx}_source_crop_id"] = str(idx)
            score_row[f"crop_{idx}_window_start_sec"] = f"{float(idx):.8f}"
        score_rows.append(score_row)

    by_group: dict[tuple[str, ...], list[int]] = {}
    for idx, row in enumerate(predictions):
        by_group.setdefault(_prediction_group_key(row, metadata_cols), []).append(idx)
    for group, indices in by_group.items():
        if len(indices) != 8:
            raise ValueError(f"score_fusion predictions require 8 trials per group: {'|'.join(group)}")
        top4 = subject_top4([float(predictions[idx]["y_score"]) for idx in indices])
        for local_idx, row_idx in enumerate(indices):
            predictions[row_idx]["pred_top4"] = str(int(top4[local_idx]))

    pred_fields = ["route_id", *metadata_cols, "subject_id", "trial_id", "y_score", "y_pred"]
    if any("y_true" in row for row in predictions):
        pred_fields.append("y_true")
    pred_fields.append("pred_top4")
    _write_csv(output_dir / "predictions.csv", predictions, pred_fields)

    score_fields = [*metadata_cols, "subject_id", "trial_id", "y_true"]
    for idx in range(5):
        score_fields.extend([f"crop_{idx}", f"crop_{idx}_source_crop_id", f"crop_{idx}_window_start_sec"])
    _write_csv(output_dir / "score_matrix.csv", score_rows, score_fields)
    return score_matrix_evidence


def _copy_manifest_artifact(source_manifest_path: Path, source_field: str, output_path: Path) -> Path | None:
    try:
        manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    source = _resolve_run_artifact(source_manifest_path, manifest.get(source_field))
    if source is None or not source.exists():
        return None
    output_path.write_bytes(source.read_bytes())
    return output_path


def _load_source_manifest(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _write_derived_dataset_manifest(
    route_data: dict[str, Any],
    output_dir: Path,
    score_matrix_path: Path,
    component_score_paths: dict[str, Path],
) -> Path:
    rows = _read_csv(score_matrix_path)
    subjects = sorted({str(row["subject_id"]) for row in rows})
    trial_index = [
        {
            "subject_id": str(row["subject_id"]),
            "trial_id": str(row["trial_id"]),
            "crop_ids": [str(idx) for idx in range(5)],
            "window_start_secs": [float(idx) for idx in range(5)],
        }
        for row in rows
    ]
    data_sources = []
    checksum_manifest = []
    for component_id, path in sorted(component_score_paths.items()):
        display = _display_path(path)
        data_sources.append({"path": display, "kind": "component_score_csv", "component_id": component_id})
        checksum_manifest.append({"path": display, "sha256": sha256_file(path)})
    dataset = {
        "dataset_version": str(route_data["dataset_version"]),
        "status": "ready",
        "description": "Derived score-fusion dataset evidence from component score CSV inputs.",
        "label_scope": {
            "train": "available",
            "val": "available",
            "test": "available_for_audit_only",
            "pseudo_public": "hidden_until_audit",
        },
        "subject_ids": subjects,
        "n_subjects": len(subjects),
        "n_trials": len(rows),
        "n_crops": 5,
        "data_sources": data_sources,
        "checksum_manifest": checksum_manifest,
        "trial_index": trial_index,
    }
    dataset_path = output_dir / "dataset_manifest.yaml"
    dataset_path.write_text(yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return dataset_path


def _write_derived_dataset_split(
    route_data: dict[str, Any],
    output_dir: Path,
    score_matrix_path: Path,
    component_score_paths: dict[str, Path],
) -> tuple[Path, Path]:
    rows = _read_csv(score_matrix_path)
    dataset_path = _write_derived_dataset_manifest(route_data, output_dir, score_matrix_path, component_score_paths)
    subjects = sorted({str(row["subject_id"]) for row in rows})
    split_rows = [
        {
            "subject_id": str(row["subject_id"]),
            "trial_id": str(row["trial_id"]),
            "original_trial_id": f"{row['subject_id']}::{row['trial_id']}",
            "split": "test",
        }
        for row in rows
    ]
    split = {
        "split_id": str(route_data["split_id"]),
        "subject_group_split": True,
        "status": "derived_component_scores_only",
        "description": "Score-fusion split evidence derived from component scores; source component manifests should be preferred for candidate audit.",
        "train_subjects": [],
        "val_subjects": [],
        "test_subjects": subjects,
        "trial_rows": split_rows,
    }
    split_path = output_dir / "split_manifest.yaml"
    split_path.write_text(yaml.safe_dump(split, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return dataset_path, split_path


def _source_split_path(source_manifest_path: Path) -> Path | None:
    manifest = _load_source_manifest(source_manifest_path)
    if manifest is None:
        return None
    source = _resolve_run_artifact(source_manifest_path, manifest.get("split_manifest_path"))
    if source is not None and source.exists():
        return source
    return None


def _unique_source_split_hashes(source_manifest_paths: list[Path]) -> set[str]:
    hashes: set[str] = set()
    for manifest_path in source_manifest_paths:
        manifest = _load_source_manifest(manifest_path)
        if manifest is not None and isinstance(manifest.get("split_sha256"), str):
            hashes.add(str(manifest["split_sha256"]))
            continue
        split_path = _source_split_path(manifest_path)
        if split_path is not None:
            hashes.add(sha256_file(split_path))
    return hashes


def _source_split_membership_hash(split_data: dict[str, Any]) -> str:
    rows = []
    for row in split_data.get("trial_rows") or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "subject_id": str(row.get("subject_id", "")),
                "trial_id": str(row.get("trial_id", row.get("original_trial_id", ""))),
                "original_trial_id": str(row.get("original_trial_id", "")),
                "split": str(row.get("split", "")),
            }
        )
    payload = {
        "train_subjects": sorted(map(str, split_data.get("train_subjects") or [])),
        "val_subjects": sorted(map(str, split_data.get("val_subjects") or [])),
        "test_subjects": sorted(map(str, split_data.get("test_subjects") or [])),
        "trial_rows": sorted(rows, key=lambda item: (item["split"], item["subject_id"], item["trial_id"], item["original_trial_id"])),
    }
    import hashlib

    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _write_merged_source_split(
    route_data: dict[str, Any],
    output_dir: Path,
    source_manifest_paths: list[Path],
) -> Path | None:
    fold_definitions: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    used_fold_ids: set[str] = set()
    seen_alignment_keys: set[tuple[str, str, str]] = set()
    for index, manifest_path in enumerate(source_manifest_paths):
        manifest = _load_source_manifest(manifest_path)
        split_path = _source_split_path(manifest_path)
        if manifest is None or split_path is None:
            continue
        try:
            split_data = yaml.safe_load(split_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(split_data, dict):
            continue
        seed = manifest.get("seed")
        fold = manifest.get("fold")
        membership_hash = _source_split_membership_hash(split_data)
        alignment_key = (str(seed), str(fold), membership_hash)
        if alignment_key in seen_alignment_keys:
            continue
        seen_alignment_keys.add(alignment_key)
        base_fold_id = f"seed{seed}_fold{fold}" if seed is not None and fold is not None else f"source{index}"
        fold_id = base_fold_id
        suffix = 1
        while fold_id in used_fold_ids:
            suffix += 1
            fold_id = f"{base_fold_id}_{suffix}"
        used_fold_ids.add(fold_id)
        fold_definitions.append(
            {
                "fold": fold_id,
                "source_manifest": _display_path(manifest_path),
                "source_split_id": str(split_data.get("split_id") or manifest.get("split_id") or ""),
                "source_split_membership_sha256": membership_hash,
                "train_subjects": list(split_data.get("train_subjects") or []),
                "val_subjects": list(split_data.get("val_subjects") or []),
                "test_subjects": list(split_data.get("test_subjects") or []),
            }
        )
        for row in split_data.get("trial_rows") or []:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            item["fold"] = fold_id
            trial_rows.append(item)
    if not fold_definitions or not trial_rows:
        return None
    split = {
        "split_id": f"{route_data['split_id']}__score_fusion_sources",
        "subject_group_split": True,
        "status": "merged_source_protocol_splits",
        "description": "Merged split evidence for score-fusion output assembled from multiple source job manifests.",
        "fold_definitions": fold_definitions,
        "trial_rows": trial_rows,
    }
    split_path = output_dir / "split_manifest.yaml"
    split_path.write_text(yaml.safe_dump(split, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return split_path


def _write_score_fusion_manifest(
    route_path: Path,
    output_dir: Path,
    component_score_paths: dict[str, Path],
    *,
    source_manifest_paths: list[Path],
    score_matrix_evidence: str,
) -> Path:
    route_data = load_route(route_path)
    route_id = str(route_data.get("route_id") or route_path.stem)
    seed = int(route_data.get("seed", 42))
    predictions_path = output_dir / "predictions.csv"
    score_matrix_path = output_dir / "score_matrix.csv"

    dataset_path: Path | None = None
    split_path: Path | None = None
    unique_split_hashes = _unique_source_split_hashes(source_manifest_paths)
    if len(source_manifest_paths) > 1 and len(unique_split_hashes) > 1:
        dataset_path = _write_derived_dataset_manifest(route_data, output_dir, score_matrix_path, component_score_paths)
        split_path = _write_merged_source_split(route_data, output_dir, source_manifest_paths)
    else:
        for source_manifest in source_manifest_paths:
            if dataset_path is None:
                dataset_path = _copy_manifest_artifact(source_manifest, "dataset_manifest_path", output_dir / "dataset_manifest.yaml")
            if split_path is None:
                split_path = _copy_manifest_artifact(source_manifest, "split_manifest_path", output_dir / "split_manifest.yaml")
            if dataset_path is not None and split_path is not None:
                break
    if dataset_path is None or split_path is None:
        dataset_path, split_path = _write_derived_dataset_split(route_data, output_dir, score_matrix_path, component_score_paths)

    metric_group_keys = tuple([*_metadata_columns(_read_csv(predictions_path)), "subject_id"])
    metric_report = build_metric_report(
        route_id=route_id,
        prediction_csv=predictions_path,
        score_matrix_csv=score_matrix_path,
        primary_metric=str(route_data["evaluation"]["primary_metric"]),
        metric_group_keys=metric_group_keys,
    )
    write_metric_report(metric_report, output_dir)

    context = prepare_run_manifest_context(
        route_config_path=route_path,
        run_dir=output_dir,
        prediction_csv=predictions_path,
        dataset_manifest_path=dataset_path,
        split_manifest_path=split_path,
        split_id=str((yaml.safe_load(split_path.read_text(encoding="utf-8")) or {}).get("split_id") or route_data["split_id"]),
        source_split_id=str(route_data["split_id"]),
        seed=seed,
        source_seed=seed,
    )
    determinism = reproducibility_manifest(ReproducibilityConfig(seed=seed))
    determinism["pythonhashseed_env"] = str(seed)
    manifest = build_run_manifest_payload(
        context,
        metrics=metric_report["metrics"],
        command=[
            "python",
            "scripts/run_score_fusion_routes.py",
            "--route-filter",
            route_id,
        ],
        score_matrix_csv=score_matrix_path,
        determinism=determinism,
        top4_group_keys=metric_group_keys,
        metric_group_keys=metric_group_keys,
    )
    score_route = score_route_by_id(route_id)
    manifest["score_matrix_evidence"] = score_matrix_evidence
    manifest["score_fusion"] = {
        "score_node": str(route_data.get("model", {}).get("score_node", "")),
        "method": score_route.method if score_route else "",
        "components": list(route_data.get("model", {}).get("components", [])),
        "weights": list(score_route.weights) if score_route and score_route.weights else None,
        "query_component": score_route.query_component if score_route else None,
        "context_components": list(score_route.context_components) if score_route else [],
        "alpha": score_route.alpha if score_route else None,
        "temperature": score_route.temperature if score_route else None,
    }
    manifest["component_score_inputs"] = [
        {
            "component_id": comp_id,
            "path": _display_path(path),
            "sha256": sha256_file(path),
        }
        for comp_id, path in sorted(component_score_paths.items())
    ]
    if source_manifest_paths:
        manifest["component_source_manifests"] = [
            {
                "path": _display_path(path),
                "sha256": sha256_file(path),
            }
            for path in source_manifest_paths
        ]
    manifest["score_fusion_source_evidence"] = {
        "source_manifest_count": len(source_manifest_paths),
        "unique_source_split_count": len(unique_split_hashes),
        "split_evidence": "merged_source_protocol_splits" if len(unique_split_hashes) > 1 else ("copied_source_split" if source_manifest_paths else "derived_component_scores_only"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_dir / "manifest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Orchestrate score_fusion route assembly.")
    parser.add_argument("--dry-run", action="store_true", help="Print score_fusion routes and exit.")
    parser.add_argument("--component-scores-dir", type=Path, help="Directory of pre-exported component score CSVs.")
    parser.add_argument("--base-runs-dir", type=Path, help="Directory containing per-route run dirs with predictions.csv.")
    parser.add_argument("--export-missing", action="store_true", help="Auto-export component scores from base route predictions.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "score_fusion_batch")
    parser.add_argument("--route-filter", help="Substring filter for score_fusion route_id.")
    parser.add_argument("--audit", action="store_true", help="Run repo_doctor experiment audit after assembly.")
    args = parser.parse_args(argv)

    routes = score_fusion_routes()
    if args.route_filter:
        routes = [(p, d) for p, d in routes if args.route_filter in str(d.get("route_id", ""))]

    if not routes:
        print("No score_fusion routes found.")
        return 0

    if args.dry_run:
        print(f"Found {len(routes)} score_fusion routes:\n")
        for route_path, data in routes:
            route_id = data["route_id"]
            score_route = score_route_by_id(str(route_id))
            components = list(data.get("model", {}).get("components", []))
            print(f"  {route_id}")
            for comp in components:
                base = base_route_for_component(str(comp))
                status = f"-> {base}" if base else "-> (no torch route)"
                print(f"    {comp} {status}")
            if score_route and score_route.weights:
                print(f"    weights: {score_route.weights}")
            print()
        return 0

    if not args.component_scores_dir and not (args.base_runs_dir and args.export_missing):
        parser.error("--component-scores-dir is required (or use --base-runs-dir --export-missing)")

    component_scores_dir = args.component_scores_dir or (args.output_dir / "component_scores")
    component_scores_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    passed = 0
    failed = 0
    exported_component_sources: dict[str, list[Path]] = {}

    for idx, (route_path, data) in enumerate(routes, 1):
        route_id = str(data["route_id"])
        components = [str(c) for c in data.get("model", {}).get("components", [])]
        print(f"[{idx}/{len(routes)}] {route_id}")

        component_scores: dict[str, Path] = {}
        source_manifest_paths: list[Path] = []
        untrusted_component_scores: list[str] = []
        missing: list[str] = []

        for comp_id in components:
            expected = component_scores_dir / f"{comp_id}.csv"
            base_route = base_route_for_component(comp_id)
            if expected.exists():
                component_scores[comp_id] = expected
                current_run_sources = exported_component_sources.get(comp_id)
                if current_run_sources:
                    for manifest_path in current_run_sources:
                        if manifest_path not in source_manifest_paths:
                            source_manifest_paths.append(manifest_path)
                else:
                    untrusted_component_scores.append(comp_id)
                continue

            # Try to export from base route predictions
            if base_route is None:
                print(f"  {comp_id}: no base route mapping (skip)")
                missing.append(comp_id)
                continue

            if args.export_missing and args.base_runs_dir:
                base_run_dir = args.base_runs_dir / base_route
                source_artifacts = find_component_source_artifacts(base_run_dir)
                if not source_artifacts:
                    print(f"  {comp_id}: no score_matrix/predictions found in {base_run_dir}")
                    missing.append(comp_id)
                    continue
                rc = export_component_score(comp_id, base_route, source_artifacts, expected)
                if rc == 0:
                    component_scores[comp_id] = expected
                    exported_component_sources[comp_id] = [
                        source.manifest_path
                        for source in source_artifacts
                        if source.manifest_path is not None
                    ]
                    for source in source_artifacts:
                        if source.manifest_path is not None and source.manifest_path not in source_manifest_paths:
                            source_manifest_paths.append(source.manifest_path)
                    source_label = ", ".join(_display_path(source.path) for source in source_artifacts[:3])
                    suffix = "" if len(source_artifacts) <= 3 else f" (+{len(source_artifacts) - 3} more)"
                    print(f"  {comp_id}: exported from {source_label}{suffix}")
                else:
                    missing.append(comp_id)
            else:
                print(f"  {comp_id}: component score not found at {expected}")
                missing.append(comp_id)

        if missing:
            print(f"  SKIP: missing component scores: {missing}")
            results.append({"route_id": route_id, "status": "SKIP", "missing_components": missing})
            failed += 1
            continue

        output_dir = args.output_dir / route_id
        rc, assembly_info = assemble_score_fusion(route_path, component_scores, output_dir)
        result: dict[str, Any] = {
            "route_id": route_id,
            "run_dir": str(output_dir),
            "status": "PASS" if rc == 0 else "FAIL",
        }

        if rc == 0:
            _write_score_fusion_manifest(
                route_path,
                output_dir,
                component_scores,
                source_manifest_paths=source_manifest_paths,
                score_matrix_evidence=str(assembly_info.get("score_matrix_evidence") or "synthetic"),
            )

        if rc == 0 and args.audit and not source_manifest_paths:
            result["status"] = "AUDIT_BLOCKED_NO_SOURCE_MANIFEST"
            result["reason"] = "--audit requires source run manifests; pre-exported component CSVs only produce diagnostic artifacts"

        if rc == 0 and args.audit and source_manifest_paths and untrusted_component_scores:
            result["status"] = "AUDIT_BLOCKED_UNTRUSTED_COMPONENT_CSV"
            result["reason"] = f"--audit requires source manifests for every component; pre-existing CSVs were not exported in this run: {untrusted_component_scores}"

        if rc == 0 and args.audit and source_manifest_paths and not untrusted_component_scores:
            audit_proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "repo_doctor.py"),
                    "experiment",
                    "--route", str(route_path),
                    "--run", str(output_dir),
                    "--gate", "candidate",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            result["audit_returncode"] = audit_proc.returncode
            result["status"] = "PASS" if audit_proc.returncode == 0 else "AUDIT_FAIL"

        if result["status"] == "PASS":
            passed += 1
            print(f"  PASS  ->  {_display_path(output_dir)}")
        else:
            failed += 1
            print(f"  {result['status']}")

        results.append(result)

    summary_path = args.output_dir / "score_fusion_summary.json"
    summary = {
        "total": len(routes),
        "passed": passed,
        "failed": failed,
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nScore fusion summary: {passed} passed, {failed} failed  ->  {_display_path(summary_path)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
