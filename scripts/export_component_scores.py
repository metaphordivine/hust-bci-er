"""Export component-score CSV tables from a base route's predictions.

Reads a torch_classifier route's ``predictions.csv`` from a run directory
and rewrites it as a component-score table suitable for
``assemble_score_route.py``.  The output adds a ``component_id`` column
and reorders columns to match the expected component-score schema.

Usage::

    python scripts/export_component_scores.py \
        --route configs/routes/models/fixed_crop_ea_fbstcnet.yaml \
        --predictions outputs/fixed_crop_ea_fbstcnet/.../predictions.csv \
        --component-id fixed_crop_ea_fbstcnet_component \
        --output outputs/component_scores/fixed_crop_ea_fbstcnet_component.csv

If ``--predictions``/``--score-matrix`` are omitted the script prefers
``<run_dir>/score_matrix.csv`` and falls back to ``<run_dir>/predictions.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from hust_bci_er.contracts.prediction import prediction_schema  # noqa: E402


def load_route(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"route config must be a mapping: {path}")
    return data


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_score_matrix(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def score_matrix_crop_columns(fields: set[str]) -> list[str]:
    cols = [f"crop_{idx}" for idx in range(5)]
    if all(col in fields for col in cols):
        return cols
    alt = [f"crop{idx}" for idx in range(5)]
    if all(col in fields for col in alt):
        return alt
    return []


def export_component_scores_from_score_matrix(
    score_matrix_path: Path,
    component_id: str,
    output_path: Path,
) -> int:
    if not score_matrix_path.exists():
        print(f"score matrix file not found: {score_matrix_path}", file=sys.stderr)
        return 1

    rows = read_score_matrix(score_matrix_path)
    if not rows:
        print(f"score matrix file is empty: {score_matrix_path}", file=sys.stderr)
        return 1

    fields = set(rows[0])
    schema = prediction_schema(fields)
    subject_col = schema["subject_id"]
    trial_col = schema["trial_id"]
    truth_col = schema["y_true"]
    crop_cols = score_matrix_crop_columns(fields)
    if subject_col is None or trial_col is None or truth_col is None or len(crop_cols) != 5:
        print(
            f"score matrix must have subject_id, trial_id, y_true, and crop_0..crop_4 columns: {score_matrix_path}",
            file=sys.stderr,
        )
        return 1

    optional_keys = []
    for col in ("seed", "fold"):
        if col in rows[0] and any(row.get(col) not in {None, ""} for row in rows):
            optional_keys.append(col)

    provenance_columns = [
        col
        for crop_col in crop_cols
        for col in (f"{crop_col}_source_crop_id", f"{crop_col}_window_start_sec")
    ]
    present_provenance_columns = [col for col in provenance_columns if col in fields]
    has_crop_provenance = bool(present_provenance_columns)
    if has_crop_provenance and len(present_provenance_columns) != len(provenance_columns):
        print(f"score matrix has partial crop provenance columns: {score_matrix_path}", file=sys.stderr)
        return 1
    provenance_fields = ["source_crop_id", "window_start_sec"] if has_crop_provenance else []

    fieldnames = ["component_id", *optional_keys, "subject_id", "trial_id", "crop_id", *provenance_fields, "score", "y_true"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for crop_id, crop_col in enumerate(crop_cols):
                item: dict[str, str] = {
                    "component_id": component_id,
                    "subject_id": str(row[subject_col]),
                    "trial_id": str(row[trial_col]),
                    "crop_id": str(crop_id),
                    "score": str(row[crop_col]),
                    "y_true": str(row[truth_col]),
                }
                for col in optional_keys:
                    item[col] = str(row.get(col, ""))
                if has_crop_provenance:
                    source_crop_id = row.get(f"{crop_col}_source_crop_id")
                    window_start_sec = row.get(f"{crop_col}_window_start_sec")
                    if source_crop_id in {None, ""} or window_start_sec in {None, ""}:
                        print(f"score matrix row has partial crop provenance values: {score_matrix_path}", file=sys.stderr)
                        return 1
                    item["source_crop_id"] = str(source_crop_id)
                    item["window_start_sec"] = str(window_start_sec)
                writer.writerow(item)
    return 0


def export_component_scores(
    predictions_path: Path,
    component_id: str,
    output_path: Path,
    *,
    route_id: str | None = None,
) -> int:
    if not predictions_path.exists():
        print(f"predictions file not found: {predictions_path}", file=sys.stderr)
        return 1

    rows = read_predictions(predictions_path)
    if not rows:
        print(f"predictions file is empty: {predictions_path}", file=sys.stderr)
        return 1

    schema = prediction_schema(set(rows[0]))
    subject_col = schema["subject_id"]
    trial_col = schema["trial_id"]
    score_col = schema["score"]

    if subject_col is None or trial_col is None or score_col is None:
        print(
            f"predictions file must have subject_id, trial_id, and score columns: {predictions_path}",
            file=sys.stderr,
        )
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine optional alignment-key columns present in the predictions table.
    optional_keys = []
    for col in ("seed", "fold", "crop_id"):
        if col in rows[0] and rows[0].get(col) not in {None, ""}:
            optional_keys.append(col)

    truth_col = schema.get("y_true")
    has_truth = truth_col is not None
    provenance_keys = ("source_crop_id", "window_start_sec")
    present_provenance_columns = [col for col in provenance_keys if col in rows[0]]
    has_provenance = bool(present_provenance_columns)
    provenance_cols: list[str] = []
    if has_provenance:
        if len(present_provenance_columns) != len(provenance_keys):
            missing_columns = [col for col in provenance_keys if col not in rows[0]]
            print(f"predictions file has partial crop provenance columns {missing_columns}: {predictions_path}", file=sys.stderr)
            return 1
        for idx, row in enumerate(rows):
            missing_values = [col for col in provenance_keys if row.get(col) in {None, ""}]
            if missing_values:
                print(f"predictions row {idx} has partial crop provenance values {missing_values}: {predictions_path}", file=sys.stderr)
                return 1
        provenance_cols = list(provenance_keys)

    fieldnames = ["component_id", *optional_keys, "subject_id", "trial_id", *provenance_cols, "score"]
    if has_truth:
        fieldnames.append("y_true")

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            item: dict[str, str] = {
                "component_id": component_id,
                "subject_id": str(row[subject_col]),
                "trial_id": str(row[trial_col]),
                "score": str(row[score_col]),
            }
            for col in optional_keys:
                item[col] = str(row.get(col, ""))
            for col in provenance_cols:
                item[col] = str(row.get(col, ""))
            if has_truth:
                item["y_true"] = str(row[truth_col])
            writer.writerow(item)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a component-score CSV from a route's predictions.")
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="Path to predictions.csv. Auto-detected from --run-dir if omitted.")
    parser.add_argument("--score-matrix", type=Path, help="Path to score_matrix.csv. Preferred for exact-metric score-fusion components.")
    parser.add_argument("--run-dir", type=Path, help="Run directory containing predictions.csv.")
    parser.add_argument("--component-id", required=True, help="Logical component ID for the score table.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    args = parser.parse_args(argv)

    score_matrix_path = args.score_matrix
    predictions_path = args.predictions
    if score_matrix_path is None and predictions_path is None and args.run_dir is not None:
        candidate_score_matrix = args.run_dir / "score_matrix.csv"
        if candidate_score_matrix.exists():
            score_matrix_path = candidate_score_matrix
        else:
            predictions_path = args.run_dir / "predictions.csv"
    if score_matrix_path is None and predictions_path is None:
        parser.error("--score-matrix, --predictions, or --run-dir is required")

    if score_matrix_path is not None:
        return export_component_scores_from_score_matrix(
            score_matrix_path,
            args.component_id,
            args.output,
        )

    return export_component_scores(
        # argparse validation above guarantees this is set here.
        predictions_path,
        args.component_id,
        args.output,
        route_id=str(load_route(args.route).get("route_id") or ""),
    )


if __name__ == "__main__":
    raise SystemExit(main())
