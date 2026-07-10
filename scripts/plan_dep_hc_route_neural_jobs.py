"""Materialize DEP/HC neural diagnostic jobs from existing route configs.

This is a planner only. It converts a route's preprocessing and model backbone
into the independent DEP/HC classifier task; it does not execute training and
does not create candidate evidence for the emotion route board.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.config.schema import preprocessing_component_name, sliding_window_count  # noqa: E402
from hust_bci_er.tasks.dep_hc.neural import DEP_HC_NEURAL_MODELS  # noqa: E402


FIELDS = [
    "run_id",
    "protocol",
    "fold",
    "holdout_seed",
    "outer_fold",
    "inner_fold",
    "outer_folds",
    "inner_folds",
    "outer_seed",
    "inner_seed",
    "seed",
    "model",
    "epochs",
    "batch_size",
    "threshold_objective",
    "subject_aggregation",
    "model_kwargs_json",
    "source_trial_sec",
    "window_sec",
    "stride_sec",
    "n_crops",
    "preprocessing",
]


@dataclass(frozen=True)
class RouteJob:
    run_id: str
    protocol: str
    fold: int
    holdout_seed: int
    seed: int
    model: str
    epochs: int
    batch_size: int
    threshold_objective: str
    subject_aggregation: str
    model_kwargs_json: str
    source_trial_sec: float
    window_sec: float
    stride_sec: float | None
    n_crops: int
    preprocessing: str
    outer_fold: int | None = None
    inner_fold: int | None = None
    outer_folds: int | None = None
    inner_folds: int | None = None
    outer_seed: int | None = None
    inner_seed: int | None = None

    def row(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "protocol": self.protocol,
            "fold": str(self.fold),
            "holdout_seed": str(self.holdout_seed),
            "outer_fold": _format_optional_int(self.outer_fold),
            "inner_fold": _format_optional_int(self.inner_fold),
            "outer_folds": _format_optional_int(self.outer_folds),
            "inner_folds": _format_optional_int(self.inner_folds),
            "outer_seed": _format_optional_int(self.outer_seed),
            "inner_seed": _format_optional_int(self.inner_seed),
            "seed": str(self.seed),
            "model": self.model,
            "epochs": str(self.epochs),
            "batch_size": str(self.batch_size),
            "threshold_objective": self.threshold_objective,
            "subject_aggregation": self.subject_aggregation,
            "model_kwargs_json": self.model_kwargs_json,
            "source_trial_sec": _format_float(self.source_trial_sec),
            "window_sec": _format_float(self.window_sec),
            "stride_sec": _format_optional_float(self.stride_sec),
            "n_crops": str(self.n_crops),
            "preprocessing": self.preprocessing,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan DEP/HC neural diagnostic jobs from route configs.")
    parser.add_argument("--route-config", type=Path, action="append", required=True, help="Route config YAML. Can be repeated.")
    parser.add_argument("--out", type=Path, required=True, help="Output TSV path.")
    parser.add_argument("--run-prefix", default="dep_hc_route", help="Prefix for generated run IDs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p1-fold", type=int, action="append", default=None, help="P1 fold to include. Repeatable.")
    parser.add_argument("--p2-holdout-seed", type=int, action="append", default=None, help="P2 holdout seed. Repeatable.")
    parser.add_argument("--p3-outer-fold", type=int, action="append", default=None, help="P3 outer fold to include. Repeatable.")
    parser.add_argument("--p3-inner-folds", type=int, default=3)
    parser.add_argument("--p3-outer-folds", type=int, default=5)
    parser.add_argument("--outer-seed", type=int, default=42)
    parser.add_argument("--inner-seed", type=int, default=123)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--batch-size-override", type=int, default=None)
    parser.add_argument("--threshold-objective", action="append", default=None)
    parser.add_argument("--subject-aggregation", action="append", default=None)
    parser.add_argument(
        "--drop-preprocessing",
        action="append",
        default=["euclidean_alignment"],
        help=(
            "Preprocessing step to omit from converted DEP/HC jobs. Can be repeated. "
            "Defaults to dropping euclidean_alignment because subject-type diagnostics do not use EA."
        ),
    )
    args = parser.parse_args(argv)

    jobs: list[RouteJob] = []
    for route_config in args.route_config:
        jobs.extend(
            plan_jobs_for_route(
                route_config,
                run_prefix=args.run_prefix,
                seed=args.seed,
                p1_folds=args.p1_fold or [0, 3],
                p2_holdout_seeds=args.p2_holdout_seed or [123, 666, 999],
                p3_outer_folds=args.p3_outer_fold,
                p3_inner_folds=args.p3_inner_folds,
                p3_outer_fold_count=args.p3_outer_folds,
                outer_seed=args.outer_seed,
                inner_seed=args.inner_seed,
                epochs_override=args.epochs_override,
                batch_size_override=args.batch_size_override,
                threshold_objective=args.threshold_objective or ["balanced_accuracy"],
                subject_aggregation=args.subject_aggregation or ["mean"],
                drop_preprocessing=set(args.drop_preprocessing or []),
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out, jobs)
    print(json.dumps({"out": str(args.out), "n_jobs": len(jobs)}, ensure_ascii=False))
    return 0


def plan_jobs_for_route(
    route_config: Path,
    *,
    run_prefix: str,
    seed: int,
    p1_folds: list[int],
    p2_holdout_seeds: list[int],
    p3_outer_folds: list[int] | None = None,
    p3_inner_folds: int = 3,
    p3_outer_fold_count: int = 5,
    outer_seed: int = 42,
    inner_seed: int = 123,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    threshold_objective: str | list[str] = "balanced_accuracy",
    subject_aggregation: str | list[str] = "mean",
    drop_preprocessing: set[str] | None = None,
) -> list[RouteJob]:
    route_data = yaml.safe_load(route_config.read_text(encoding="utf-8")) or {}
    if not isinstance(route_data, dict):
        raise ValueError(f"route config must be a mapping: {route_config}")
    route_id = str(route_data.get("route_id") or route_config.stem)
    model_data = route_data.get("model") or {}
    if not isinstance(model_data, dict):
        raise ValueError(f"route model must be a mapping: {route_config}")
    model_name = str(model_data.get("name") or "")
    if model_name not in DEP_HC_NEURAL_MODELS:
        raise ValueError(f"route {route_id} uses unsupported DEP/HC neural model: {model_name}")
    model_kwargs = {str(k): v for k, v in model_data.items() if k != "name"}
    preprocessing = route_data.get("preprocessing") or []
    if not isinstance(preprocessing, list):
        raise ValueError(f"route preprocessing must be a list: {route_config}")
    drop_preprocessing = {"euclidean_alignment"} if drop_preprocessing is None else set(drop_preprocessing)
    converted_preprocessing = [
        step
        for step in preprocessing
        if str(preprocessing_component_name(step) or step) not in drop_preprocessing
    ]
    preprocessing_token = _serialize_preprocessing(converted_preprocessing)
    training = route_data.get("training") or {}
    if not isinstance(training, dict):
        training = {}
    source_trial_sec, window_sec, stride_sec, n_crops = _route_window_plan(route_data)
    epochs = int(epochs_override if epochs_override is not None else training.get("epochs", 5))
    batch_size = int(batch_size_override if batch_size_override is not None else training.get("batch_size", 32))
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    model_kwargs_json = json.dumps(model_kwargs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    route_slug = _slug(route_id)
    threshold_objectives = _as_list(threshold_objective)
    subject_aggregations = _as_list(subject_aggregation)
    combos = [(objective, aggregation) for objective in threshold_objectives for aggregation in subject_aggregations]
    multi_combo = len(combos) > 1

    jobs: list[RouteJob] = []
    for threshold_item, aggregation_item in combos:
        combo_suffix = f"_thr{_slug(threshold_item)}_agg{_slug(aggregation_item)}" if multi_combo else ""
        for holdout_seed in p2_holdout_seeds:
            jobs.append(
                RouteJob(
                    run_id=f"{run_prefix}_{route_slug}{combo_suffix}_p2_h{int(holdout_seed)}",
                    protocol="p2",
                    fold=0,
                    holdout_seed=int(holdout_seed),
                    outer_fold=None,
                    inner_fold=None,
                    outer_folds=None,
                    inner_folds=None,
                    outer_seed=None,
                    inner_seed=None,
                    seed=int(seed),
                    model=model_name,
                    epochs=epochs,
                    batch_size=batch_size,
                    threshold_objective=threshold_item,
                    subject_aggregation=aggregation_item,
                    model_kwargs_json=model_kwargs_json,
                    source_trial_sec=source_trial_sec,
                    window_sec=window_sec,
                    stride_sec=stride_sec,
                    n_crops=n_crops,
                    preprocessing=preprocessing_token,
                )
            )
        for fold in p1_folds:
            jobs.append(
                RouteJob(
                    run_id=f"{run_prefix}_{route_slug}{combo_suffix}_p1_f{int(fold)}",
                    protocol="p1",
                    fold=int(fold),
                    holdout_seed=0,
                    outer_fold=None,
                    inner_fold=None,
                    outer_folds=None,
                    inner_folds=None,
                    outer_seed=None,
                    inner_seed=None,
                    seed=int(seed),
                    model=model_name,
                    epochs=epochs,
                    batch_size=batch_size,
                    threshold_objective=threshold_item,
                    subject_aggregation=aggregation_item,
                    model_kwargs_json=model_kwargs_json,
                    source_trial_sec=source_trial_sec,
                    window_sec=window_sec,
                    stride_sec=stride_sec,
                    n_crops=n_crops,
                    preprocessing=preprocessing_token,
                )
            )
        for outer_fold in p3_outer_folds or []:
            for inner_fold in range(int(p3_inner_folds)):
                jobs.append(
                    RouteJob(
                        run_id=f"{run_prefix}_{route_slug}{combo_suffix}_p3_o{int(outer_fold)}_i{int(inner_fold)}",
                        protocol="p3",
                        fold=0,
                        holdout_seed=0,
                        outer_fold=int(outer_fold),
                        inner_fold=int(inner_fold),
                        outer_folds=int(p3_outer_fold_count),
                        inner_folds=int(p3_inner_folds),
                        outer_seed=int(outer_seed),
                        inner_seed=int(inner_seed),
                        seed=int(inner_seed) + int(outer_fold) * 1000 + int(inner_fold) * 100,
                        model=model_name,
                        epochs=epochs,
                        batch_size=batch_size,
                        threshold_objective=threshold_item,
                        subject_aggregation=aggregation_item,
                        model_kwargs_json=model_kwargs_json,
                        source_trial_sec=source_trial_sec,
                        window_sec=window_sec,
                        stride_sec=stride_sec,
                        n_crops=n_crops,
                        preprocessing=preprocessing_token,
                    )
                )
            jobs.append(
                RouteJob(
                    run_id=f"{run_prefix}_{route_slug}{combo_suffix}_p3_o{int(outer_fold)}_final",
                    protocol="p3",
                    fold=0,
                    holdout_seed=0,
                    outer_fold=int(outer_fold),
                    inner_fold=None,
                    outer_folds=int(p3_outer_fold_count),
                    inner_folds=int(p3_inner_folds),
                    outer_seed=int(outer_seed),
                    inner_seed=int(inner_seed),
                    seed=int(outer_seed) + int(outer_fold),
                    model=model_name,
                    epochs=epochs,
                    batch_size=batch_size,
                    threshold_objective=threshold_item,
                    subject_aggregation=aggregation_item,
                    model_kwargs_json=model_kwargs_json,
                    source_trial_sec=source_trial_sec,
                    window_sec=window_sec,
                    stride_sec=stride_sec,
                    n_crops=n_crops,
                    preprocessing=preprocessing_token,
                )
            )
    return jobs


def _route_window_plan(route_data: dict[str, Any]) -> tuple[float, float, float | None, int]:
    augmentation = route_data.get("augmentation") or {}
    if not isinstance(augmentation, dict):
        augmentation = {}
    source_trial_sec = float(augmentation.get("source_trial_sec", 50.0))
    window_sec = float(augmentation.get("window_sec", route_data.get("input_window_sec", 10.0)))
    stride_sec: float | None = None
    if "n_crops" in augmentation:
        n_crops = int(augmentation["n_crops"])
    elif "stride_sec" in augmentation:
        stride_sec = float(augmentation["stride_sec"])
        if stride_sec <= 0:
            raise ValueError("sliding-window route stride_sec must be positive")
        n_crops = sliding_window_count(source_trial_sec, window_sec, stride_sec)
    else:
        n_crops = 5
    if source_trial_sec <= 0:
        raise ValueError("source_trial_sec must be positive")
    if window_sec <= 0:
        raise ValueError("window_sec must be positive")
    if n_crops <= 0:
        raise ValueError("n_crops must be positive")
    if window_sec > source_trial_sec:
        raise ValueError("window_sec cannot exceed source_trial_sec")
    return source_trial_sec, window_sec, stride_sec, n_crops


def write_tsv(path: Path, jobs: list[RouteJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(FIELDS)]
    for job in jobs:
        row = job.row()
        lines.append("\t".join(_tsv_cell(row[field]) for field in FIELDS))
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def _tsv_cell(value: str) -> str:
    if "\t" in value or "\n" in value or "\r" in value:
        raise ValueError(f"TSV cell contains a forbidden control character: {value!r}")
    return value


def _serialize_preprocessing(preprocessing: list[Any]) -> str:
    if not preprocessing:
        return "none"
    if len(preprocessing) == 1 and isinstance(preprocessing[0], str):
        return preprocessing[0]
    return json.dumps(preprocessing, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _as_list(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        out = [str(item) for item in value]
    else:
        out = [str(value)]
    if not out or any(not item for item in out):
        raise ValueError("planner sweep values must be non-empty")
    return out


def _format_float(value: float) -> str:
    return f"{float(value):g}"


def _format_optional_float(value: float | None) -> str:
    return "default" if value is None else _format_float(value)


def _format_optional_int(value: int | None) -> str:
    return "" if value is None else str(int(value))


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


if __name__ == "__main__":
    raise SystemExit(main())
