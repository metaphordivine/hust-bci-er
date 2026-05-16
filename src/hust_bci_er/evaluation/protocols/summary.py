"""Summaries for completed protocol runner jobs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prediction_jobs(protocol_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(job)
        for job in protocol_manifest.get("jobs", [])
        if isinstance(job, Mapping) and "predictions.csv" in (job.get("expected_artifacts") or [])
    ]


def completed_job_rows(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    protocol_manifest = _read_json(run_dir / "protocol_run_manifest.json")
    job_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for job in prediction_jobs(protocol_manifest):
        job_id = str(job["job_id"])
        job_run = run_dir / "job_runs" / job_id
        manifest_path = job_run / "manifest.json"
        audit_path = job_run / "audit_report.json"
        subject_path = job_run / "subject_ba.csv"
        if not manifest_path.exists():
            missing.append(f"{job_id}: manifest.json")
            continue
        manifest = _read_json(manifest_path)
        metric = str(manifest.get("primary_metric") or "")
        metrics = manifest.get("metrics") if isinstance(manifest.get("metrics"), dict) else {}
        value = metrics.get(metric)
        audit_overall = ""
        if audit_path.exists():
            audit_overall = str(_read_json(audit_path).get("overall") or "")
        job_rows.append(
            {
                "job_id": job_id,
                "route_id": str(job["route_id"]),
                "seed": int(job["seed"]),
                "fold": "" if job.get("fold") is None else int(job["fold"]),
                "stage": str(job["stage"]),
                "split_id": str(job["split_id"]),
                "run_dir": str(job_run),
                "primary_metric": metric,
                "metric_value": "" if value is None else float(value),
                "audit_overall": audit_overall,
            }
        )
        if subject_path.exists():
            for row in _read_csv(subject_path):
                subject_rows.append(
                    {
                        "job_id": job_id,
                        "route_id": str(job["route_id"]),
                        "seed": str(job["seed"]),
                        "fold": "" if job.get("fold") is None else str(job["fold"]),
                        **row,
                    }
                )
    return job_rows, subject_rows, missing


def aggregate_board_rows(job_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in job_rows:
        value = row.get("metric_value")
        if value == "":
            continue
        grouped[str(row["route_id"])].append(float(value))
    out: list[dict[str, Any]] = []
    for route_id, values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        out.append(
            {
                "route_id": route_id,
                "n_jobs": int(arr.size),
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        )
    return out


def write_protocol_summary(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    job_rows, subject_rows, missing = completed_job_rows(run_dir)
    board_rows = aggregate_board_rows(job_rows)

    board_path = run_dir / "protocol1_board.csv"
    with board_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["route_id", "n_jobs", "mean", "std", "min", "max"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(board_rows)

    jobs_path = run_dir / "protocol1_jobs.csv"
    with jobs_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["job_id", "route_id", "seed", "fold", "stage", "split_id", "run_dir", "primary_metric", "metric_value", "audit_overall"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(job_rows)

    subject_path = run_dir / "protocol1_subject_ba.csv"
    with subject_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["job_id", "route_id", "seed", "fold", "subject_id", "balanced_accuracy", "metric_value", "n_rows", "group_key"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(subject_rows)

    audit = {
        "status": "COMPLETE" if not missing else "INCOMPLETE",
        "n_prediction_jobs": len(prediction_jobs(_read_json(run_dir / "protocol_run_manifest.json"))),
        "n_completed_jobs": len(job_rows),
        "missing_artifacts": missing,
        "board": board_path.as_posix(),
        "jobs": jobs_path.as_posix(),
        "subject_ba": subject_path.as_posix(),
    }
    audit_path = run_dir / "protocol1_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit
