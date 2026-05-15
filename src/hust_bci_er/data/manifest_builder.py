"""Build auditable dataset manifests from a raw index table."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from hust_bci_er.audit.manifest import sha256_file


REQUIRED_INDEX_COLUMNS = {"path", "subject_id", "trial_id"}


def read_index_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_INDEX_COLUMNS - fields)
        if missing:
            raise ValueError(f"index CSV missing columns: {', '.join(missing)}")
        return [dict(row) for row in reader]


def normalize_bool(value: Any) -> bool | None:
    if value in {None, ""}:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def resolve_data_path(raw_path: str, *, base_dir: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path)


def trial_index_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    base_dir: Path,
    require_files: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str]]:
    trial_rows: list[dict[str, Any]] = []
    checksums: list[dict[str, str]] = []
    missing_sources: list[str] = []
    seen_paths: set[str] = set()
    for idx, row in enumerate(rows):
        rel_path = str(row.get("path") or "").replace("\\", "/")
        subject_id = str(row.get("subject_id") or "")
        trial_id = str(row.get("trial_id") or "")
        if not rel_path or not subject_id or not trial_id:
            raise ValueError(f"row {idx} must include path, subject_id, and trial_id")
        resolved = resolve_data_path(rel_path, base_dir=base_dir)
        if require_files and not resolved.exists():
            raise FileNotFoundError(f"data source not found: {resolved}")
        if not resolved.exists() and rel_path not in missing_sources:
            missing_sources.append(rel_path)
        item: dict[str, Any] = {
            "path": rel_path,
            "subject_id": subject_id,
            "trial_id": trial_id,
        }
        for optional in ["crop_id", "split", "sampling_rate_hz", "n_channels", "n_samples"]:
            value = row.get(optional)
            if value not in {None, ""}:
                item[optional] = int(value) if optional in {"crop_id", "n_channels", "n_samples"} else value
        if row.get("y_true") not in {None, ""}:
            item["y_true"] = int(row["y_true"])
            item["label_available"] = True
        else:
            item["label_available"] = bool(normalize_bool(row.get("label_available")) or False)
        trial_rows.append(item)
        if rel_path not in seen_paths:
            seen_paths.add(rel_path)
            if resolved.exists():
                checksums.append({"path": rel_path, "sha256": sha256_file(resolved)})
    return trial_rows, checksums, sorted(missing_sources)


def build_dataset_manifest(
    *,
    dataset_version: str,
    rows: Iterable[Mapping[str, Any]],
    base_dir: Path,
    description: str = "",
    require_files: bool = True,
    label_scope: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    trial_rows, checksums, missing_sources = trial_index_rows(rows, base_dir=base_dir, require_files=require_files)
    checksum_paths = {item["path"] for item in checksums}
    sources = [{"path": str(row["path"]), "kind": "file", "checksum_available": str(row["path"]) in checksum_paths} for row in trial_rows]
    unique_sources = [dict(item) for item in {item["path"]: item for item in sources}.values()]
    subjects = sorted({str(row["subject_id"]) for row in trial_rows})
    trials = sorted({(str(row["subject_id"]), str(row["trial_id"])) for row in trial_rows})
    crops = sorted({str(row.get("crop_id")) for row in trial_rows if row.get("crop_id") is not None})
    if not trial_rows:
        status = "declared_without_raw_data_index"
    elif missing_sources:
        status = "declared_with_missing_data_sources"
    else:
        status = "ready"
    return {
        "dataset_version": dataset_version,
        "status": status,
        "description": description or "Built from raw data index.",
        "label_scope": dict(
            label_scope
            or {
                "train": "available",
                "val": "available",
                "test": "available_for_audit_only",
                "pseudo_public": "hidden_until_audit",
            }
        ),
        "subject_ids": subjects,
        "n_subjects": len(subjects),
        "n_trials": len(trials),
        "n_crops": len(crops),
        "missing_data_sources": missing_sources,
        "data_sources": unique_sources,
        "checksum_manifest": checksums,
        "trial_index": trial_rows,
    }


def write_dataset_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(manifest), sort_keys=False, allow_unicode=True), encoding="utf-8")
