"""Run a small neural sanity baseline for independent DEP/HC classification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.tasks.dep_hc.experiment import write_dep_hc_task_outputs  # noqa: E402
from hust_bci_er.tasks.dep_hc.neural import evaluate_dep_hc_neural_task  # noqa: E402
from hust_bci_er.training import _real_adapter_impl as real_adapter  # noqa: E402
from scripts.run_dep_hc_router import _make_samples, _subjects_for_protocol, resolve_preprocessing  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a DEP/HC neural sanity baseline.")
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
    parser.add_argument("--preprocessing", action="append", default=None)
    parser.add_argument("--model-name", choices=["eegnet", "shallow_conv_net"], default="eegnet")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-mode", choices=["balanced", "uniform"], default="balanced")
    parser.add_argument("--threshold-objective", choices=["balanced_accuracy", "min_recall", "fixed_0_5"], default="balanced_accuracy")
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
    result = evaluate_dep_hc_neural_task(
        train_samples,
        eval_samples,
        val_samples=val_samples,
        model_name=args.model_name,
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weight_mode=args.class_weight_mode,
        threshold_objective=args.threshold_objective,
    )
    config = {
        "task": "dep_hc_neural",
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
        "model_name": args.model_name,
        "class_weight_mode": args.class_weight_mode,
        "threshold_objective": args.threshold_objective,
        "label_source": "cohort field; subject/trial identifiers are split and audit metadata only",
        "train_subjects": sorted(train_subjects),
        "val_subjects": sorted(val_subjects),
        "test_subjects": sorted(test_subjects),
    }
    write_dep_hc_task_outputs(result, args.out_dir, config=config)
    print(json.dumps({"out_dir": str(args.out_dir.resolve()), "metrics": result["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
