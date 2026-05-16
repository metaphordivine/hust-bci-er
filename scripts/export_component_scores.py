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

If ``--predictions`` is omitted the script looks for
``<run_dir>/predictions.csv``.
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

    fieldnames = ["component_id", *optional_keys, "subject_id", "trial_id", "score"]
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
            if has_truth:
                item["y_true"] = str(row[truth_col])
            writer.writerow(item)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a component-score CSV from a route's predictions.")
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="Path to predictions.csv. Auto-detected from --run-dir if omitted.")
    parser.add_argument("--run-dir", type=Path, help="Run directory containing predictions.csv.")
    parser.add_argument("--component-id", required=True, help="Logical component ID for the score table.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    args = parser.parse_args(argv)

    predictions_path = args.predictions
    if predictions_path is None and args.run_dir is not None:
        predictions_path = args.run_dir / "predictions.csv"
    if predictions_path is None:
        parser.error("--predictions or --run-dir is required")

    return export_component_scores(
        predictions_path,
        args.component_id,
        args.output,
        route_id=str(load_route(args.route).get("route_id") or ""),
    )


if __name__ == "__main__":
    raise SystemExit(main())
