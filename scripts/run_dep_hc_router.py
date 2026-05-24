"""Run a DEP/HC subject-kind router diagnostic.

This is a separate cohort-classification task, not an emotion Top-4 route.
It trains on EEG-derived features and reports held-out-subject DEP/HC
classification metrics plus calibrated probabilities for later soft routing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.analysis.dep_hc_router import RouterSample, evaluate_router, write_router_outputs  # noqa: E402
from hust_bci_er.evaluation.protocols.subject_splits import p1_subject_split, p2_subject_split  # noqa: E402
from hust_bci_er.training import _real_adapter_impl as real_adapter  # noqa: E402


DEFAULT_PREPROCESSING: tuple[str, ...] = ("car", "zscore")


def resolve_preprocessing(values: list[str] | None) -> list[str]:
    return list(values) if values is not None else list(DEFAULT_PREPROCESSING)


def _subjects_for_protocol(
    trial_rows: list[dict[str, Any]],
    *,
    protocol: str,
    seed: int,
    fold: int,
    n_folds: int,
    n_holdout_subjects: int,
    holdout_seed: int,
) -> tuple[set[str], set[str], set[str], str]:
    if protocol == "p1":
        train, val, test = p1_subject_split(trial_rows, seed=seed, fold=fold, n_folds=n_folds)
        split_id = f"dep_hc_router_p1_seed{seed}_fold{fold}"
    elif protocol == "p2":
        train, val, test = p2_subject_split(
            trial_rows,
            holdout_seed=holdout_seed,
            train_seed=seed,
            n_holdout_subjects=n_holdout_subjects,
        )
        split_id = f"dep_hc_router_p2_holdout{holdout_seed}_train{seed}"
    else:
        raise ValueError(f"unsupported protocol: {protocol}")
    return set(train), set(val), set(test), split_id


def _make_samples(
    trials: list[dict[str, Any]],
    *,
    preprocessing: list[str],
    source_trial_sec: float,
    window_sec: float,
    n_crops: int,
    stride_sec: float | None = None,
    ea_transform,
) -> list[RouterSample]:
    if stride_sec is None:
        windows = real_adapter._make_fixed_crops(
            trials,
            source_trial_sec=source_trial_sec,
            window_sec=window_sec,
            n_crops=n_crops,
            preproc=preprocessing,
            ea_transform=ea_transform,
            skip_preproc=False,
        )
    else:
        windows = real_adapter._make_sliding_windows(
            trials,
            source_trial_sec=source_trial_sec,
            window_sec=window_sec,
            stride_sec=stride_sec,
            preproc=preprocessing,
            ea_transform=ea_transform,
            skip_preproc=False,
        )
    return [
        RouterSample(
            x=window["x"],
            subject_id=str(window["subject_id"]),
            trial_id=str(window["trial_id"]),
            crop_id=int(window["crop_id"]),
            window_start_sec=float(window["window_start_sec"]),
            cohort=str(window["cohort"]),
        )
        for window in windows
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate a DEP/HC subject-kind router diagnostic.")
    parser.add_argument("--data-root", type=Path, help="HUST EEG .mat data root.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=["p1", "p2"], default="p1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-holdout-subjects", type=int, default=12)
    parser.add_argument("--holdout-seed", type=int, default=999)
    parser.add_argument("--source-trial-sec", type=float, default=50.0)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--n-crops", type=int, default=5)
    parser.add_argument("--preprocessing", action="append", default=None, help="Preprocessing step. Can be repeated. Overrides the default car+zscore pipeline when supplied.")
    parser.add_argument("--feature-set", choices=["cov_tangent", "bandpower", "cov_tangent_bandpower"], default="cov_tangent_bandpower")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--l2", type=float, default=1e-3)
    args = parser.parse_args(argv)

    preprocessing = resolve_preprocessing(args.preprocessing)
    real_adapter._validate_adapter_preprocessing(preprocessing)

    data_root = real_adapter._resolve_data_root(args.data_root)
    trials = real_adapter._load_mat_trials(data_root)
    trial_rows = [
        {"subject_id": str(trial["subject_id"]), "trial_id": str(trial["trial_id"]), "cohort": str(trial["cohort"])}
        for trial in trials
    ]
    train_subjects, val_subjects, test_subjects, split_id = _subjects_for_protocol(
        trial_rows,
        protocol=args.protocol,
        seed=args.seed,
        fold=args.fold,
        n_folds=args.n_folds,
        n_holdout_subjects=args.n_holdout_subjects,
        holdout_seed=args.holdout_seed,
    )

    train_trials = [trial for trial in trials if trial["subject_id"] in train_subjects]
    val_trials = [trial for trial in trials if trial["subject_id"] in val_subjects]
    eval_trials = [trial for trial in trials if trial["subject_id"] in test_subjects]
    raw_train = real_adapter._make_fixed_crops(
        train_trials,
        source_trial_sec=args.source_trial_sec,
        window_sec=args.window_sec,
        n_crops=args.n_crops,
        preproc=preprocessing,
        skip_preproc=True,
    )
    ea_transform = real_adapter._fit_ea_on_windows(raw_train, preprocessing)
    train_samples = _make_samples(
        train_trials,
        preprocessing=preprocessing,
        source_trial_sec=args.source_trial_sec,
        window_sec=args.window_sec,
        n_crops=args.n_crops,
        ea_transform=ea_transform,
    )
    val_samples = _make_samples(
        val_trials,
        preprocessing=preprocessing,
        source_trial_sec=args.source_trial_sec,
        window_sec=args.window_sec,
        n_crops=args.n_crops,
        ea_transform=ea_transform,
    )
    eval_samples = _make_samples(
        eval_trials,
        preprocessing=preprocessing,
        source_trial_sec=args.source_trial_sec,
        window_sec=args.window_sec,
        n_crops=args.n_crops,
        ea_transform=ea_transform,
    )
    result = evaluate_router(
        train_samples,
        eval_samples,
        val_samples=val_samples,
        feature_set=args.feature_set,
        lr=args.lr,
        epochs=args.epochs,
        l2=args.l2,
    )
    config = {
        "protocol": args.protocol,
        "split_id": split_id,
        "seed": args.seed,
        "fold": args.fold,
        "n_folds": args.n_folds,
        "n_holdout_subjects": args.n_holdout_subjects,
        "holdout_seed": args.holdout_seed,
        "source_trial_sec": args.source_trial_sec,
        "window_sec": args.window_sec,
        "n_crops": args.n_crops,
        "preprocessing": preprocessing,
        "feature_set": args.feature_set,
        "train_subjects": sorted(train_subjects),
        "val_subjects": sorted(val_subjects),
        "test_subjects": sorted(test_subjects),
        "label_source": "cohort field derived from dataset manifest/subject prefix; not used as input feature",
    }
    write_router_outputs(result, args.out_dir, config=config)
    print(json.dumps({"out_dir": str(args.out_dir.resolve()), "metrics": result["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
