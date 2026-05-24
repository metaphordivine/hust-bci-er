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
    parser.add_argument("--protocol", choices=["p1", "p2", "p3"], default=None)
    parser.add_argument("--split-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--n-holdout-subjects", type=int, default=None)
    parser.add_argument("--holdout-seed", type=int, default=None)
    parser.add_argument("--outer-fold", type=int, default=None)
    parser.add_argument("--inner-fold", type=int, default=None)
    parser.add_argument("--outer-folds", type=int, default=None)
    parser.add_argument("--inner-folds", type=int, default=None)
    parser.add_argument("--outer-seed", type=int, default=None)
    parser.add_argument("--inner-seed", type=int, default=None)
    parser.add_argument("--eval-scope", default=None)
    args = parser.parse_args(argv)

    component_roots = _named_roots(args.component_run_dir)
    selection_component_roots = _named_roots(args.selection_component_run_dir or [])
    components = _read_named_prediction_dirs(component_roots)
    selection_components = (
        _read_named_prediction_dirs(selection_component_roots)
        if selection_component_roots
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
    split_metadata = _split_metadata_for_output(args, component_roots)
    config = {
        "task": "dep_hc_score_fusion",
        **split_metadata,
        "component_roots": _string_roots(component_roots),
        "selection_component_roots": _string_roots(selection_component_roots),
        "threshold_objective": args.threshold_objective,
        "subject_aggregation": args.subject_aggregation,
        "weight_step": args.weight_step,
        "select_two_way_weight": args.select_two_way_weight,
        "label_source": "component prediction rows; subject/trial identifiers are alignment and audit metadata only",
    }
    write_dep_hc_task_outputs(result, args.out_dir, config=config)
    print(json.dumps({"out_dir": str(args.out_dir.resolve()), "metrics": result["metrics"]}, ensure_ascii=False))
    return 0


def _read_named_prediction_dirs(roots: dict[str, Path]) -> dict[str, list[dict[str, str]]]:
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


def _string_roots(roots: dict[str, Path]) -> dict[str, str]:
    return {name: str(path) for name, path in roots.items()}


def _split_metadata_for_output(args: argparse.Namespace, component_roots: dict[str, Path]) -> dict[str, object]:
    metadata, missing_diagnostics = _common_component_metadata(component_roots)
    if missing_diagnostics:
        missing_args = _missing_explicit_split_args(args)
        if missing_args:
            missing_components = "; ".join(missing_diagnostics)
            missing_options = ", ".join(f"--{name.replace('_', '-')}" for name in missing_args)
            raise SystemExit(
                "score fusion component runs are missing dep_hc_task_diagnostic.json: "
                f"{missing_components}. Provide complete explicit split metadata including {missing_options}."
            )
        metadata = {}
    overrides = {
        "protocol": args.protocol,
        "split_id": args.split_id,
        "seed": args.seed,
        "fold": args.fold,
        "n_folds": args.n_folds,
        "n_holdout_subjects": args.n_holdout_subjects,
        "holdout_seed": args.holdout_seed,
        "outer_fold": args.outer_fold,
        "inner_fold": args.inner_fold,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "outer_seed": args.outer_seed,
        "inner_seed": args.inner_seed,
        "eval_scope": args.eval_scope,
    }
    metadata.update({key: value for key, value in overrides.items() if value is not None})
    if not metadata.get("protocol"):
        raise SystemExit(
            "score fusion requires split metadata; provide --protocol/--split-id or use component runs "
            "with dep_hc_task_diagnostic.json"
        )
    return metadata


def _common_component_metadata(component_roots: dict[str, Path]) -> tuple[dict[str, object], list[str]]:
    keys = (
        "protocol",
        "split_id",
        "seed",
        "fold",
        "n_folds",
        "n_holdout_subjects",
        "holdout_seed",
        "outer_fold",
        "inner_fold",
        "outer_folds",
        "inner_folds",
        "outer_seed",
        "inner_seed",
        "eval_scope",
    )
    required_alignment_keys = {
        "protocol",
        "split_id",
        "n_folds",
        "n_holdout_subjects",
        "holdout_seed",
        "outer_fold",
        "inner_fold",
        "outer_folds",
        "inner_folds",
        "outer_seed",
        "inner_seed",
        "eval_scope",
    }
    values_by_key: dict[str, set[object]] = {key: set() for key in keys}
    missing_diagnostics: list[str] = []
    for name, root in component_roots.items():
        path = root / "dep_hc_task_diagnostic.json"
        if not path.exists():
            missing_diagnostics.append(f"{name}={path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        if not isinstance(config, dict):
            raise SystemExit(f"component diagnostic config must be an object: {name}={path}")
        for key in keys:
            value = _metadata_value(key, config.get(key))
            if value in {"", None}:
                continue
            values_by_key[key].add(value)
    metadata: dict[str, object] = {}
    for key, values in values_by_key.items():
        if len(values) > 1:
            if key in required_alignment_keys:
                raise SystemExit(f"component split metadata mismatch for {key}: {sorted(map(str, values))}")
            continue
        if values:
            metadata[key] = next(iter(values))
    return metadata, missing_diagnostics


def _missing_explicit_split_args(args: argparse.Namespace) -> list[str]:
    required = ["protocol", "split_id"]
    if args.protocol == "p1":
        required.extend(["seed", "fold", "n_folds"])
    elif args.protocol == "p2":
        required.extend(["holdout_seed", "n_holdout_subjects"])
    elif args.protocol == "p3":
        required.extend(["eval_scope", "outer_fold", "outer_folds", "outer_seed"])
        if args.eval_scope == "p3_inner_validation" or args.inner_fold is not None:
            required.extend(["inner_fold", "inner_folds", "inner_seed"])
    return [name for name in required if getattr(args, name) is None]


def _metadata_value(key: str, value: object) -> object:
    if value is None or value == "":
        return None
    if key in {
        "seed",
        "fold",
        "n_folds",
        "n_holdout_subjects",
        "holdout_seed",
        "outer_fold",
        "inner_fold",
        "outer_folds",
        "inner_folds",
        "outer_seed",
        "inner_seed",
    }:
        return int(value)
    return str(value)


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
