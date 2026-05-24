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

from hust_bci_er.tasks.dep_hc.neural import DEP_HC_NEURAL_MODELS  # noqa: E402


FIELDS = [
    "run_id",
    "protocol",
    "fold",
    "holdout_seed",
    "seed",
    "model",
    "epochs",
    "batch_size",
    "threshold_objective",
    "subject_aggregation",
    "model_kwargs_json",
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
    preprocessing: str

    def row(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "protocol": self.protocol,
            "fold": str(self.fold),
            "holdout_seed": str(self.holdout_seed),
            "seed": str(self.seed),
            "model": self.model,
            "epochs": str(self.epochs),
            "batch_size": str(self.batch_size),
            "threshold_objective": self.threshold_objective,
            "subject_aggregation": self.subject_aggregation,
            "model_kwargs_json": self.model_kwargs_json,
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
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--batch-size-override", type=int, default=None)
    parser.add_argument("--threshold-objective", default="balanced_accuracy")
    parser.add_argument("--subject-aggregation", default="mean")
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
                epochs_override=args.epochs_override,
                batch_size_override=args.batch_size_override,
                threshold_objective=args.threshold_objective,
                subject_aggregation=args.subject_aggregation,
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
    epochs_override: int | None,
    batch_size_override: int | None,
    threshold_objective: str,
    subject_aggregation: str,
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
    converted_preprocessing = [str(step) for step in preprocessing if str(step) not in drop_preprocessing]
    preprocessing_token = ",".join(converted_preprocessing) if converted_preprocessing else "default"
    training = route_data.get("training") or {}
    if not isinstance(training, dict):
        training = {}
    epochs = int(epochs_override if epochs_override is not None else training.get("epochs", 5))
    batch_size = int(batch_size_override if batch_size_override is not None else training.get("batch_size", 32))
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    model_kwargs_json = json.dumps(model_kwargs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    route_slug = _slug(route_id)

    jobs: list[RouteJob] = []
    for holdout_seed in p2_holdout_seeds:
        jobs.append(
            RouteJob(
                run_id=f"{run_prefix}_{route_slug}_p2_h{int(holdout_seed)}",
                protocol="p2",
                fold=0,
                holdout_seed=int(holdout_seed),
                seed=int(seed),
                model=model_name,
                epochs=epochs,
                batch_size=batch_size,
                threshold_objective=threshold_objective,
                subject_aggregation=subject_aggregation,
                model_kwargs_json=model_kwargs_json,
                preprocessing=preprocessing_token,
            )
        )
    for fold in p1_folds:
        jobs.append(
            RouteJob(
                run_id=f"{run_prefix}_{route_slug}_p1_f{int(fold)}",
                protocol="p1",
                fold=int(fold),
                holdout_seed=0,
                seed=int(seed),
                model=model_name,
                epochs=epochs,
                batch_size=batch_size,
                threshold_objective=threshold_objective,
                subject_aggregation=subject_aggregation,
                model_kwargs_json=model_kwargs_json,
                preprocessing=preprocessing_token,
            )
        )
    return jobs


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


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


if __name__ == "__main__":
    raise SystemExit(main())
