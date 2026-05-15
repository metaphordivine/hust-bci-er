"""Deterministic toy route adapter for end-to-end audit smoke runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from hust_bci_er.audit.manifest import sha256_file
from hust_bci_er.audit.run_manifest import write_run_manifest
from hust_bci_er.contracts.records import PredictionRecord
from hust_bci_er.data.csv_loader import CsvManifestLoader
from hust_bci_er.evaluation.prediction_writer import write_predictions
from hust_bci_er.evaluation.report import build_metric_report, write_metric_report


SUBJECTS = {
    "toy_train": "train",
    "toy_test": "test",
}


@dataclass(frozen=True)
class ToyRunArtifacts:
    run_dir: Path
    dataset_manifest: Path
    split_manifest: Path
    prediction_csv: Path
    score_matrix_csv: Path
    manifest_json: Path
    metric_report: Mapping[str, Any]


def label_for_trial(index: int) -> int:
    return 1 if index < 4 else 0


def feature_for_trial(*, subject_index: int, trial_index: int, crop_index: int = 0) -> tuple[float, float]:
    label = label_for_trial(trial_index)
    polarity = 1.0 if label else -1.0
    return (
        polarity * (2.0 + 0.05 * subject_index + 0.01 * crop_index),
        polarity * (0.5 + 0.02 * trial_index),
    )


def write_feature_csv(path: Path, values: tuple[float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature_0", "feature_1"])
        writer.writeheader()
        writer.writerow({"feature_0": f"{values[0]:.8f}", "feature_1": f"{values[1]:.8f}"})


def write_toy_evidence(run_dir: Path, *, dataset_version: str, split_id: str) -> tuple[Path, Path]:
    data_dir = run_dir / "data"
    rows: list[dict[str, Any]] = []
    checksums: list[dict[str, str]] = []
    for subject_index, (subject_id, split) in enumerate(SUBJECTS.items()):
        for trial_index in range(8):
            trial_id = f"trial_{trial_index}"
            rel_path = f"data/{subject_id}_{trial_id}.csv"
            path = run_dir / rel_path
            write_feature_csv(path, feature_for_trial(subject_index=subject_index, trial_index=trial_index))
            row = {
                "path": rel_path,
                "subject_id": subject_id,
                "trial_id": trial_id,
                "crop_id": 0,
                "split": split,
                "sampling_rate_hz": 128,
                "n_channels": 2,
                "n_samples": 1280,
                "y_true": label_for_trial(trial_index),
                "label_available": True,
            }
            rows.append(row)
            checksums.append({"path": rel_path, "sha256": sha256_file(path)})
    dataset_manifest = {
        "dataset_version": dataset_version,
        "status": "ready",
        "description": "Synthetic two-subject EEG-like toy dataset for CI-only end-to-end audit smoke.",
        "label_scope": {
            "train": "available",
            "val": "available",
            "test": "available_for_audit_only",
            "pseudo_public": "hidden_until_audit",
        },
        "subject_ids": sorted(SUBJECTS),
        "n_subjects": len(SUBJECTS),
        "n_trials": len(rows),
        "n_crops": 1,
        "missing_data_sources": [],
        "data_sources": [{"path": item["path"], "kind": "file", "checksum_available": True} for item in rows],
        "checksum_manifest": checksums,
        "trial_index": rows,
    }
    dataset_path = run_dir / "dataset_manifest.yaml"
    dataset_path.write_text(yaml.safe_dump(dataset_manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")

    split_rows = [
        {"subject_id": row["subject_id"], "original_trial_id": f"{row['subject_id']}::{row['trial_id']}", "split": row["split"]}
        for row in rows
    ]
    split_manifest = {
        "split_id": split_id,
        "subject_group_split": True,
        "status": "ready",
        "description": "Synthetic subject holdout split for toy CI candidate audit smoke.",
        "train_subjects": ["toy_train"],
        "val_subjects": [],
        "test_subjects": ["toy_test"],
        "trial_rows": split_rows,
    }
    split_path = run_dir / "split_manifest.yaml"
    split_path.write_text(yaml.safe_dump(split_manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return dataset_path, split_path


def load_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML document must be a mapping: {path}")
    return data


def model_name(route_data: Mapping[str, Any]) -> str:
    model = route_data.get("model")
    if isinstance(model, Mapping):
        return str(model.get("name") or "")
    return str(model or "")


def fit_centroid(train_x: np.ndarray, train_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if set(np.unique(train_y).tolist()) != {0, 1}:
        raise ValueError("toy adapter requires both binary classes in the train split")
    negative = train_x[train_y == 0].mean(axis=0)
    positive = train_x[train_y == 1].mean(axis=0)
    return negative, positive


def centroid_score(x: np.ndarray, *, negative: np.ndarray, positive: np.ndarray) -> np.ndarray:
    distance_to_negative = np.linalg.norm(x - negative, axis=1)
    distance_to_positive = np.linalg.norm(x - positive, axis=1)
    return distance_to_negative - distance_to_positive


def write_score_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["subject_id", "trial_id", "y_true", "crop_0", "crop_1", "crop_2", "crop_3", "crop_4"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_toy_route(
    *,
    route_config_path: Path,
    run_dir: Path,
    split_id: str | None = None,
    seed: int | None = None,
    command: str | list[str] = "python scripts/train_route.py",
    **__,
) -> ToyRunArtifacts:
    route_config_path = route_config_path.resolve()
    route_data = load_mapping(route_config_path)
    if str(route_data.get("dataset_version")) != "toy_eeg_v1":
        raise ValueError("toy adapter only supports dataset_version: toy_eeg_v1")
    if model_name(route_data) not in {"logistic_regression", "toy_centroid"}:
        raise ValueError("toy adapter supports only logistic_regression or toy_centroid route smoke configs")

    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    source_split_id = str(route_data["split_id"])
    active_split_id = split_id or source_split_id
    active_seed = int(seed if seed is not None else route_data["seed"])
    dataset_path, split_path = write_toy_evidence(run_dir, dataset_version=str(route_data["dataset_version"]), split_id=active_split_id)
    dataset_data = load_mapping(dataset_path)
    loader = CsvManifestLoader(dataset_data, base_dir=run_dir)
    train = loader.model_batch(split="train")
    test = loader.model_batch(split="test")
    if train.y is None or test.y is None:
        raise ValueError("toy adapter requires labels for train and audit-only test evaluation")
    negative, positive = fit_centroid(train.x, train.y)
    scores = centroid_score(test.x, negative=negative, positive=positive)
    y_pred = (scores >= 0).astype(int)

    records = [
        PredictionRecord(
            route_id=str(route_data["route_id"]),
            subject_id=str(test.metadata["subject_id"][idx]),
            trial_id=str(test.metadata["trial_id"][idx]),
            crop_id=0,
            y_score=float(scores[idx]),
            y_pred=int(y_pred[idx]),
            y_true=int(test.y[idx]),
            seed=active_seed,
        )
        for idx in range(len(test.x))
    ]
    prediction_path = run_dir / "predictions.csv"
    write_predictions(records, prediction_path, group_keys=("subject_id",))

    score_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        base = float(scores[idx])
        score_rows.append(
            {
                "subject_id": record.subject_id,
                "trial_id": record.trial_id,
                "y_true": int(record.y_true or 0),
                "crop_0": f"{base:.8f}",
                "crop_1": f"{base + 0.001:.8f}",
                "crop_2": f"{base + 0.002:.8f}",
                "crop_3": f"{base + 0.003:.8f}",
                "crop_4": f"{base + 0.004:.8f}",
            }
        )
    score_matrix_path = run_dir / "score_matrix.csv"
    write_score_matrix(score_matrix_path, score_rows)

    metric_report = build_metric_report(
        route_id=str(route_data["route_id"]),
        prediction_csv=prediction_path,
        score_matrix_csv=score_matrix_path,
        primary_metric=str(route_data["evaluation"]["primary_metric"]),
    )
    write_metric_report(metric_report, run_dir)
    (run_dir / "model_state.local.json").write_text(
        json.dumps(
            {
                "adapter": "toy_centroid",
                "negative_centroid": negative.tolist(),
                "positive_centroid": positive.tolist(),
                "seed": active_seed,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    write_run_manifest(
        route_config_path=route_config_path,
        run_dir=run_dir,
        prediction_csv=prediction_path,
        dataset_manifest_path=dataset_path,
        split_manifest_path=split_path,
        split_id=active_split_id,
        source_split_id=source_split_id,
        seed=active_seed,
        source_seed=int(route_data["seed"]),
        score_matrix_csv=score_matrix_path,
        metrics=metric_report["metrics"],
        command=command,
        top4_group_keys=("subject_id",),
        metric_group_keys=("subject_id",),
    )
    return ToyRunArtifacts(
        run_dir=run_dir,
        dataset_manifest=dataset_path,
        split_manifest=split_path,
        prediction_csv=prediction_path,
        score_matrix_csv=score_matrix_path,
        manifest_json=run_dir / "manifest.json",
        metric_report=metric_report,
    )
