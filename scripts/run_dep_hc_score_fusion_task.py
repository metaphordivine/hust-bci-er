"""Fuse DEP/HC probabilities from existing diagnostic run outputs.

This is diagnostic-only. It consumes compact run artifacts already produced by
DEP/HC tasks and writes a new DEP/HC diagnostic output directory. It does not
train a model and does not create emotion-route candidate evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.tasks.dep_hc.experiment import write_dep_hc_task_outputs  # noqa: E402
from hust_bci_er.tasks.dep_hc.fusion import evaluate_dep_hc_score_fusion  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DEP/HC score-level fusion from prediction CSVs.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--component-run-dir",
        action="append",
        required=True,
        help="Component as name=run_dir. The run_dir must contain dep_hc_predictions.csv.",
    )
    parser.add_argument(
        "--selection-component-run-dir",
        action="append",
        default=None,
        help="Optional validation/selection component as name=run_dir. Names must match components.",
    )
    parser.add_argument(
        "--component-weight",
        action="append",
        default=None,
        help="Fixed component weight as name=value. If omitted, equal fixed weights are used unless --select-two-way-weight is set.",
    )
    parser.add_argument("--select-two-way-weight", action="store_true")
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument(
        "--threshold-objective",
        choices=["balanced_accuracy", "min_recall", "fixed_0_5", "dep_recall_floor_0p8_hc"],
        default="fixed_0_5",
    )
    parser.add_argument(
        "--subject-aggregation",
        choices=["mean", "median", "trimmed_mean", "vote_frac"],
        default="vote_frac",
    )
    args = parser.parse_args(argv)

    components = _read_named_prediction_dirs(args.component_run_dir)
    selection_components = (
        _read_named_prediction_dirs(args.selection_component_run_dir)
        if args.selection_component_run_dir
        else None
    )
    weights = _parse_named_floats(args.component_weight)
    result = evaluate_dep_hc_score_fusion(
        components,
        val_component_prediction_rows=selection_components,
        weights=weights,
        threshold_objective=args.threshold_objective,
        subject_aggregation=args.subject_aggregation,
        select_two_way_weight=args.select_two_way_weight,
        weight_step=args.weight_step,
    )
    config = {
        "task": "dep_hc_score_fusion",
        "component_roots": _named_roots(args.component_run_dir),
        "selection_component_roots": _named_roots(args.selection_component_run_dir or []),
        "threshold_objective": args.threshold_objective,
        "subject_aggregation": args.subject_aggregation,
        "weight_step": args.weight_step,
        "select_two_way_weight": args.select_two_way_weight,
        "label_source": "component prediction rows; subject/trial identifiers are alignment and audit metadata only",
    }
    write_dep_hc_task_outputs(result, args.out_dir, config=config)
    print(json.dumps({"out_dir": str(args.out_dir.resolve()), "metrics": result["metrics"]}, ensure_ascii=False))
    return 0


def _read_named_prediction_dirs(values: list[str]) -> dict[str, list[dict[str, str]]]:
    roots = _named_roots(values)
    return {name: _read_prediction_rows(path / "dep_hc_predictions.csv") for name, path in roots.items()}


def _named_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"expected name=run_dir, got: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"component name is empty: {value}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if name in roots:
            raise SystemExit(f"duplicate component name: {name}")
        roots[name] = path
    return roots


def _read_prediction_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing component predictions: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_named_floats(values: list[str] | None) -> dict[str, float] | None:
    if not values:
        return None
    out: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"expected name=value, got: {value}")
        name, raw_number = value.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"weight name is empty: {value}")
        if name in out:
            raise SystemExit(f"duplicate weight name: {name}")
        out[name] = float(raw_number)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
