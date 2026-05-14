"""Lightweight QA summaries for dataset manifests."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


def dataset_qa(manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in manifest.get("trial_index", []) if isinstance(row, Mapping)]
    subjects = {str(row.get("subject_id")) for row in rows if row.get("subject_id") is not None}
    trials = {(str(row.get("subject_id")), str(row.get("trial_id"))) for row in rows if row.get("trial_id") is not None}
    crops = [row.get("crop_id") for row in rows if row.get("crop_id") not in {None, ""}]
    labels = Counter(str(row.get("y_true")) for row in rows if row.get("y_true") not in {None, ""})
    sources = {str(item.get("path")) for item in manifest.get("data_sources", []) if isinstance(item, Mapping)}
    checksums = {str(item.get("path")) for item in manifest.get("checksum_manifest", []) if isinstance(item, Mapping)}
    return {
        "dataset_version": manifest.get("dataset_version"),
        "n_subjects": len(subjects),
        "n_trials": len(trials),
        "n_rows": len(rows),
        "n_crop_rows": len(crops),
        "n_unique_crop_ids": len(set(crops)),
        "n_crops": len(crops),
        "label_balance": dict(sorted(labels.items())),
        "missing_checksum_paths": sorted(sources - checksums),
        "extra_checksum_paths": sorted(checksums - sources),
        "label_available_rows": sum(1 for row in rows if row.get("label_available") is True or row.get("y_true") not in {None, ""}),
    }


def render_qa_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Dataset QA Report",
        "",
        f"- dataset_version: `{report.get('dataset_version')}`",
        f"- n_subjects: `{report.get('n_subjects')}`",
        f"- n_trials: `{report.get('n_trials')}`",
        f"- n_rows: `{report.get('n_rows')}`",
        f"- n_crops: `{report.get('n_crops')}`",
        f"- n_unique_crop_ids: `{report.get('n_unique_crop_ids')}`",
        f"- label_available_rows: `{report.get('label_available_rows')}`",
        f"- label_balance: `{report.get('label_balance')}`",
        f"- missing_checksum_paths: `{report.get('missing_checksum_paths')}`",
        f"- extra_checksum_paths: `{report.get('extra_checksum_paths')}`",
        "",
    ]
    return "\n".join(lines)
