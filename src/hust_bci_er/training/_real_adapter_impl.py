"""Real EEG classifier job adapter for .mat dataset routes.

Wires route config -> .mat loading -> preprocessing -> augmentation -> torch
training -> inference -> prediction & score_matrix artifacts -> audit manifests.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from hust_bci_er.audit.manifest import sha256_file
from hust_bci_er.audit.run_manifest import (
    build_run_manifest_payload,
    prepare_run_manifest_context,
)
from hust_bci_er.config.schema import validate_route_config
from hust_bci_er.contracts.records import PredictionRecord
from hust_bci_er.data.augmentations import apply_transforms_to_windows, transform_configs_from_route
from hust_bci_er.data.windowing import fixed_crop_slices, fixed_crop_spec_from_config
from hust_bci_er.evaluation.prediction_writer import write_predictions
from hust_bci_er.evaluation.report import build_metric_report, write_metric_report
# build_model and classifier imports are lazy (inside run_real_classifier_route / _predict_scores)
# so that importing this module does not require torch to be installed.
from hust_bci_er.preprocessing.euclidean_alignment import apply_ea_transform, fit_ea_transform
from hust_bci_er.preprocessing.normalization import zscore_per_channel
from hust_bci_er.preprocessing.whitening import channel_whiten

SFREQ = 250
KEY_TO_LABEL = {"EEG_data_neu": 0, "EEG_data_pos": 1}
SOURCE_TRIALS_PER_CLASS = 4
FIXED_CANDIDATE_CROPS = 5
DATA_ROOT_ENV = "HUST_BCI_ER_DATA_ROOT"
DATA_ROOT_DEFAULT = Path(__file__).resolve().parents[3] / "scratch" / "local_data" / "hust_bci_er_train" / "训练集"


def _resolve_data_root(data_root: Path | None) -> Path:
    if data_root is not None:
        return data_root.resolve()
    import os

    env_root = os.environ.get(DATA_ROOT_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve()
    return DATA_ROOT_DEFAULT.resolve()


def _load_mat_trials(data_root: Path) -> list[dict[str, Any]]:
    """Load all .mat trials from the BCI emotion dataset."""
    trials: list[dict[str, Any]] = []
    for path in sorted(data_root.rglob("*timedata.mat")):
        subject_id = path.stem.removesuffix("timedata")
        if subject_id.startswith("DEP"):
            cohort = "DEP"
        elif subject_id.startswith("HC"):
            cohort = "HC"
        else:
            raise ValueError(f"unknown HUST EEG subject prefix in {path.name}; expected DEP* or HC*")
        for key, label in KEY_TO_LABEL.items():
            mat = _read_mat(path, key)
            samples_per_trial = mat.shape[1] // SOURCE_TRIALS_PER_CLASS
            for idx in range(SOURCE_TRIALS_PER_CLASS):
                start = idx * samples_per_trial
                stop = start + samples_per_trial
                label_name = "pos" if label == 1 else "neu"
                trials.append({
                    "x": mat[:, start:stop].astype(np.float32),
                    "y": label,
                    "subject_id": subject_id,
                    "cohort": cohort,
                    "trial_id": f"{subject_id}_{label_name}{idx + 1}",
                    "_mat_path": str(path.resolve()),
                })
    return trials


def _read_mat(path: Path, key: str) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for real HUST EEG .mat routes; install with `pip install -e \".[models]\"`."
        ) from exc
    try:
        with h5py.File(path, "r") as f:
            if key not in f:
                raise KeyError(f"{path.name} is missing dataset {key}")
            data = np.asarray(f[key][()], dtype=np.float32)
    except OSError as exc:
        raise ValueError(f"{path} is not an HDF5/v7.3 MATLAB .mat file readable by h5py") from exc
    if data.shape[0] == 30:
        return data
    if data.shape[1] == 30:
        return data.T
    raise ValueError(f"unexpected data shape in {path.name}:{key}: {data.shape}")


def _split_subjects(
    trials: list[dict[str, Any]],
    *,
    val_dep: int,
    val_hc: int,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    """Split subjects into train/val/test, mutually disjoint.

    val and test each get up to val_dep/val_hc subjects per cohort.
    train gets the rest. If there aren't enough subjects, val and test are capped.
    """
    rng = np.random.default_rng(seed)
    by_cohort: dict[str, list[str]] = {}
    for trial in trials:
        by_cohort.setdefault(trial["cohort"], []).append(trial["subject_id"])
    by_cohort = {c: sorted(set(ids)) for c, ids in by_cohort.items()}

    val_subjects: set[str] = set()
    test_subjects: set[str] = set()
    for cohort, n_val in [("DEP", val_dep), ("HC", val_hc)]:
        subjects = np.array(by_cohort.get(cohort, []), dtype=object)
        rng.shuffle(subjects)
        n_total = len(subjects)
        # ensure at least 1 train subject per cohort if possible
        n_holdout = min(n_val, (n_total - 1) // 2) if n_total >= 3 else 0
        n_test_actual = n_holdout
        val_subjects.update(map(str, subjects[:n_holdout]))
        test_subjects.update(map(str, subjects[n_holdout:n_holdout + n_test_actual]))
    train_subjects = {t["subject_id"] for t in trials} - val_subjects - test_subjects
    if not train_subjects:
        raise ValueError("training set is empty; increase smoke_n_dep/smoke_n_hc")
    return train_subjects, val_subjects, test_subjects


def _clip_trial(x: np.ndarray, duration_sec: float) -> np.ndarray:
    n_samples = int(round(duration_sec * SFREQ))
    return x[:, :min(n_samples, x.shape[1])]


KNOWN_PREPROC = frozenset({
    "zscore", "robust_zscore", "whitening_eps1e3", "whitening_eps3e4",
    "shrinkage_whitening", "euclidean_alignment", "car", "bandpass",
})


def _preprocessing_step(item: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(item, str):
        return item, {}
    if isinstance(item, Mapping):
        name = str(item.get("name") or "")
        params = {key: value for key, value in item.items() if key != "name"}
        return name, params
    return str(item), {}


def _preprocessing_names(preproc_items: Sequence[Any]) -> list[str]:
    return [_preprocessing_step(item)[0] for item in preproc_items]


def _apply_preprocessing(x: np.ndarray, preproc_names: Sequence[Any], *, ea_transform: np.ndarray | None = None) -> np.ndarray:
    """Apply a sequence of preprocessing steps to one window [channels, time]."""
    for item in preproc_names:
        name, params = _preprocessing_step(item)
        if name not in KNOWN_PREPROC:
            raise ValueError(f"unknown preprocessing step: {name}")
        if name == "zscore":
            if params:
                raise ValueError("zscore preprocessing does not accept parameters")
            x = zscore_per_channel(x)
        elif name == "robust_zscore":
            if params:
                raise ValueError("robust_zscore preprocessing does not accept parameters")
            from hust_bci_er.preprocessing.normalization import robust_zscore_per_channel
            x = robust_zscore_per_channel(x)
        elif name == "whitening_eps1e3":
            if params:
                raise ValueError("whitening_eps1e3 preprocessing does not accept parameters")
            x = channel_whiten(x, eps=1e-3)
        elif name == "whitening_eps3e4":
            if params:
                raise ValueError("whitening_eps3e4 preprocessing does not accept parameters")
            x = channel_whiten(x, eps=3e-4)
        elif name == "shrinkage_whitening":
            if params:
                raise ValueError("shrinkage_whitening preprocessing does not accept parameters")
            x = channel_whiten(x, eps=1e-3, shrinkage_alpha=0.1)
        elif name == "euclidean_alignment":
            if params:
                raise ValueError("euclidean_alignment preprocessing does not accept parameters")
            if ea_transform is not None:
                x = apply_ea_transform(x, ea_transform)
        elif name == "car":
            if params:
                raise ValueError("car preprocessing does not accept parameters")
            from hust_bci_er.preprocessing.normalization import common_average_reference
            x = common_average_reference(x)
        elif name == "bandpass":
            from hust_bci_er.preprocessing.filtering import bandpass_filter
            x = bandpass_filter(
                x,
                sfreq=SFREQ,
                low_hz=float(params.get("low_hz", 1.0)),
                high_hz=float(params.get("high_hz", min(45.0, SFREQ / 2.0 - 1.0))),
                order=int(params.get("order", 4)),
            )
    return x


def _validate_adapter_preprocessing(preproc_names: Sequence[Any]) -> None:
    unknown = [name for name in _preprocessing_names(preproc_names) if name not in KNOWN_PREPROC]
    if unknown:
        raise ValueError(f"unknown preprocessing step(s): {', '.join(sorted(set(unknown)))}")


def _make_sliding_windows(
    trials: list[dict[str, Any]],
    *,
    source_trial_sec: float,
    window_sec: float,
    stride_sec: float,
    preproc: list[str],
    ea_transform: np.ndarray | None = None,
    skip_preproc: bool = False,
) -> list[dict[str, Any]]:
    """Create sliding windows from trials (split-first: trials already filtered by split).

    When ``skip_preproc=True``, raw windows are returned for EA fitting.
    """
    source_samples = int(round(source_trial_sec * SFREQ))
    window_samples = int(round(window_sec * SFREQ))
    stride_samples = int(round(stride_sec * SFREQ))
    windows: list[dict[str, Any]] = []
    for trial in trials:
        x = _clip_trial(trial["x"].astype(np.float32), source_trial_sec)
        n_times = x.shape[1]
        crop_id = 0
        for start in range(0, n_times - window_samples + 1, stride_samples):
            win = x[:, start:start + window_samples].copy()
            if not skip_preproc:
                win = _apply_preprocessing(win, preproc, ea_transform=ea_transform)
            windows.append({
                "x": win,
                "y": trial["y"],
                "subject_id": trial["subject_id"],
                "cohort": trial["cohort"],
                "trial_id": trial["trial_id"],
                "crop_id": crop_id,
                "window_start_sec": start / SFREQ,
            })
            crop_id += 1
    return windows


def _make_single_crops(
    trials: list[dict[str, Any]],
    *,
    window_sec: float,
    preproc: list[str],
    ea_transform: np.ndarray | None = None,
    skip_preproc: bool = False,
) -> list[dict[str, Any]]:
    """Create single-crop windows (no sliding)."""
    crops: list[dict[str, Any]] = []
    for trial in trials:
        x = _clip_trial(trial["x"].astype(np.float32), window_sec)
        if not skip_preproc:
            x = _apply_preprocessing(x, preproc, ea_transform=ea_transform)
        crops.append({
            "x": x,
            "y": trial["y"],
            "subject_id": trial["subject_id"],
            "cohort": trial["cohort"],
            "trial_id": trial["trial_id"],
            "crop_id": 0,
            "window_start_sec": 0.0,
        })
    return crops


def _make_fixed_crops(
    trials: list[dict[str, Any]],
    *,
    source_trial_sec: float,
    window_sec: float,
    n_crops: int,
    preproc: list[str],
    ea_transform: np.ndarray | None = None,
    skip_preproc: bool = False,
) -> list[dict[str, Any]]:
    """Create non-overlapping fixed crops from each source trial."""
    spec = fixed_crop_spec_from_config(
        {"source_trial_sec": source_trial_sec, "window_sec": window_sec, "n_crops": n_crops}
    )
    crops: list[dict[str, Any]] = []
    for trial in trials:
        x_full = _clip_trial(trial["x"].astype(np.float32), source_trial_sec)
        for crop_id, (start, stop, start_sec) in enumerate(fixed_crop_slices(x_full.shape[1], spec)):
            x = x_full[:, start:stop].copy()
            if not skip_preproc:
                x = _apply_preprocessing(x, preproc, ea_transform=ea_transform)
            crops.append({
                "x": x,
                "y": trial["y"],
                "subject_id": trial["subject_id"],
                "cohort": trial["cohort"],
                "trial_id": trial["trial_id"],
                "crop_id": crop_id,
                "window_start_sec": start_sec,
            })
    return crops


def _validate_fixed_crop_coverage(
    trials: list[dict[str, Any]],
    *,
    source_trial_sec: float,
    window_sec: float,
    n_crops: int,
    split_name: str,
) -> None:
    required_samples = max(int(round(source_trial_sec * SFREQ)), int(round(window_sec * SFREQ)) * n_crops)
    bad_trials = [
        str(trial.get("trial_id", "<unknown>"))
        for trial in trials
        if np.asarray(trial["x"]).shape[1] < required_samples
    ]
    if bad_trials:
        raise ValueError(
            f"{split_name} fixed-crop evidence requires at least {required_samples} samples "
            f"per trial for source_trial_sec={source_trial_sec:g} and {n_crops} crops; "
            f"too short: {', '.join(bad_trials[:5])}"
        )


def _fit_ea_on_windows(train_windows: list[dict[str, Any]], preproc: list[str]) -> np.ndarray | None:
    """Fit Euclidean Alignment on training windows if EA is in the preprocessing list.

    Must be called after window creation so EA sees all training windows,
    not just the first window_sec of each trial.
    """
    if "euclidean_alignment" not in _preprocessing_names(preproc):
        return None
    return fit_ea_transform([w["x"] for w in train_windows])


class _WindowDataset:
    """Simple PyTorch-style dataset for EEG windows."""

    def __init__(self, windows: list[dict[str, Any]]) -> None:
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        w = self.windows[idx]
        return w["x"].copy(), w["y"]


def _aggregate_to_trials(
    window_rows: list[dict[str, Any]],
    *,
    method: str = "majority_vote",
    tie_break: str = "mean_score",
) -> list[dict[str, Any]]:
    """Aggregate per-window predictions to per-trial predictions."""
    from collections import defaultdict

    by_trial: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        by_trial[(row["subject_id"], row["trial_id"])].append(row)

    trials: list[dict[str, Any]] = []
    for (subject_id, trial_id), rows in sorted(by_trial.items()):
        y_true = rows[0]["y_true"]
        scores = [r["y_score"] for r in rows]

        if method == "majority_vote":
            preds = [r["y_pred"] for r in rows]
            n_pos = sum(preds)
            n_neg = len(preds) - n_pos
            if n_pos > n_neg:
                y_pred = 1
            elif n_neg > n_pos:
                y_pred = 0
            else:
                # tie: use mean_score
                y_pred = int(np.mean(scores) >= 0.5)
        else:
            y_pred = int(np.mean(scores) >= 0.5)

        y_score = float(np.mean(scores))
        trials.append({
            "subject_id": subject_id,
            "cohort": rows[0]["cohort"],
            "trial_id": trial_id,
            "y_true": y_true,
            "y_score": y_score,
            "y_pred": y_pred,
            "n_windows": len(rows),
        })
    return trials


def _predict_scores(
    model,
    windows: list[dict[str, Any]],
    *,
    batch_size: int,
    device: str,
) -> list[dict[str, Any]]:
    """Run inference and return per-window scores."""
    import torch
    from torch.utils.data import DataLoader
    from hust_bci_er.training.classifier import logits_from_output

    loader = DataLoader(_WindowDataset(windows), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    rows: list[dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device).float()
            logits = logits_from_output(model(x_batch))
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            y_np = y_batch.numpy().astype(int)
            for local_idx in range(len(probs)):
                w = windows[offset + local_idx]
                rows.append({
                    "subject_id": w["subject_id"],
                    "cohort": w["cohort"],
                    "trial_id": w["trial_id"],
                    "crop_id": w["crop_id"],
                    "window_start_sec": w["window_start_sec"],
                    "y_true": int(y_np[local_idx]),
                    "y_score": float(probs[local_idx]),
                    "y_pred": int(preds[local_idx]),
                })
            offset += len(probs)
    return rows


def _build_score_matrix(
    prediction_rows: list[dict[str, Any]],
    *,
    crop_policy: str,
    seed: int,
) -> tuple[list[dict[str, Any]], str]:
    """Build score_matrix rows with exactly 5 crops per trial.

    Returns (rows, evidence_type) where evidence_type is "genuine" if every trial
    contributed >=5 real crop/window scores, or "synthetic" if any trial fell back
    to single-score perturbation.
    """
    from collections import defaultdict

    by_trial: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        by_trial[(row["subject_id"], row["trial_id"])].append(row)

    rows: list[dict[str, Any]] = []
    all_genuine = True
    for (subject_id, trial_id), trial_rows in sorted(by_trial.items()):
        trial_rows = sorted(trial_rows, key=lambda r: r.get("crop_id", 0))
        y_true = trial_rows[0]["y_true"]
        unique_crop_rows: list[dict[str, Any]] = []
        seen_crop_ids: set[str] = set()
        seen_window_starts: set[str] = set()
        for row_item in trial_rows:
            crop_id = row_item.get("crop_id")
            window_start = row_item.get("window_start_sec")
            if crop_id is None or window_start is None:
                continue
            crop_key = str(crop_id)
            start_key = f"{float(window_start):.8f}"
            if crop_key in seen_crop_ids or start_key in seen_window_starts:
                continue
            seen_crop_ids.add(crop_key)
            seen_window_starts.add(start_key)
            unique_crop_rows.append(row_item)

        if len(unique_crop_rows) >= 5:
            crops = unique_crop_rows[:5]
            row = {
                "subject_id": subject_id,
                "trial_id": trial_id,
                "y_true": y_true,
            }
            for i in range(5):
                row[f"crop_{i}"] = f"{crops[i]['y_score']:.8f}"
                row[f"crop_{i}_source_crop_id"] = str(crops[i]["crop_id"])
                row[f"crop_{i}_window_start_sec"] = f"{float(crops[i]['window_start_sec']):.8f}"
            rows.append(row)
        else:
            all_genuine = False
            # Single crop or insufficient windows: generate 5 deterministic
            # perturbations for smoke/diagnostic compatibility. Candidate audit
            # rejects this evidence type.
            base = trial_rows[0]["y_score"]
            row = {
                "subject_id": subject_id,
                "trial_id": trial_id,
                "y_true": y_true,
            }
            rng = np.random.default_rng(seed)
            for i in range(5):
                row[f"crop_{i}"] = f"{base + float(rng.normal(0, 1e-6)):.8f}"
            rows.append(row)
    return rows, "genuine" if all_genuine else "synthetic"


def _write_evidence_manifests(
    run_dir: Path,
    *,
    route_data: Mapping[str, Any],
    active_split_id: str,
    all_trials: list[dict[str, Any]],
    train_subjects: set[str],
    val_subjects: set[str],
    test_subjects: set[str],
    data_root: Path | None = None,
    run_mode: str = "smoke",
) -> tuple[Path, Path]:
    """Write dataset_manifest.yaml and split_manifest.yaml as run-local evidence."""
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_version = str(route_data["dataset_version"])
    window_sec = float(route_data.get("input_window_sec", 10))
    aug = route_data.get("augmentation")
    if isinstance(aug, dict) and aug.get("name") == "split_first_sliding_window":
        source_samples = int(round(float(aug["source_trial_sec"]) * SFREQ))
        window_samples = int(round(float(aug["window_sec"]) * SFREQ))
        stride_samples = int(round(float(aug["stride_sec"]) * SFREQ))
        window_start_step_sec = stride_samples / SFREQ
        actual_n_crops = max(1, (source_samples - window_samples) // stride_samples + 1)
    elif isinstance(aug, dict) and aug.get("name") == "split_first_fixed_crops":
        window_sec = float(aug["window_sec"])
        window_start_step_sec = window_sec
        actual_n_crops = int(aug["n_crops"])
    elif run_mode == "candidate":
        window_start_step_sec = window_sec
        actual_n_crops = FIXED_CANDIDATE_CROPS
    else:
        window_start_step_sec = window_sec
        actual_n_crops = 1

    # Write one feature-anchor CSV per trial with channel-mean signal statistics.
    # These are not model features — they serve as lightweight checksum anchors
    # so the audit system can verify that raw data files are present and readable.
    trial_index: list[dict[str, Any]] = []
    checksums: list[dict[str, str]] = []
    subject_ids: set[str] = set()
    for trial in sorted(all_trials, key=lambda t: (t["subject_id"], t["trial_id"])):
        subject_ids.add(trial["subject_id"])
        rel = f"data/{trial['subject_id']}_{trial['trial_id']}.csv"
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        x = trial["x"].astype(np.float64)
        mean_val = float(x.mean())
        std_val = float(x.std())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["feature_0", "feature_1"])
            writer.writeheader()
            writer.writerow({"feature_0": f"{mean_val:.8f}", "feature_1": f"{std_val:.8f}"})
        split = "train" if trial["subject_id"] in train_subjects else ("val" if trial["subject_id"] in val_subjects else "test")
        trial_index.append({
            "path": rel,
            "subject_id": trial["subject_id"],
            "trial_id": trial["trial_id"],
            "crop_id": 0,
            "n_crops": actual_n_crops,
            "crop_ids": list(range(actual_n_crops)),
            "window_start_secs": [
                idx * window_start_step_sec
                for idx in range(actual_n_crops)
            ],
            "split": split,
            "sampling_rate_hz": 250,
            "n_channels": 30,
            "n_samples": int(x.shape[1]),
            "y_true": trial["y"],
            "label_available": True,
        })
        checksums.append({"path": rel, "sha256": sha256_file(path)})

    # In full/candidate modes, record original .mat paths as primary data sources.
    raw_sources: list[dict[str, Any]] = []
    if run_mode in {"full_subjects", "candidate"}:
        resolved_data_root = data_root.resolve() if data_root is not None else None
        mat_paths = sorted({t.get("_mat_path") for t in all_trials if isinstance(t.get("_mat_path"), str)})
        for mp in mat_paths:
            source_path = Path(mp)
            entry = {
                "path": str(source_path),
                "kind": "mat",
                "sha256": sha256_file(source_path),
                "exists": source_path.exists(),
            }
            if resolved_data_root is not None:
                try:
                    entry["root_relative_path"] = source_path.relative_to(resolved_data_root).as_posix()
                    entry["root_env"] = DATA_ROOT_ENV
                except ValueError:
                    pass
            raw_sources.append(entry)

    dataset_manifest = {
        "dataset_version": dataset_version,
        "status": "ready",
        "description": f"Real EEG dataset manifest for {dataset_version}, {len(trial_index)} trials across {len(subject_ids)} subjects.",
        "label_scope": {
            "train": "available",
            "val": "available",
            "test": "available_for_audit_only",
            "pseudo_public": "hidden_until_audit",
        },
        "subject_ids": sorted(subject_ids),
        "n_subjects": len(subject_ids),
        "n_trials": len(trial_index),
        "n_crops": actual_n_crops,
        "missing_data_sources": [],
        "data_sources": [{"path": item["path"], "kind": "file", "checksum_available": True} for item in trial_index],
        "checksum_manifest": checksums,
        "trial_index": trial_index,
    }
    if raw_sources:
        dataset_manifest["raw_data_sources"] = raw_sources
    dataset_path = run_dir / "dataset_manifest.yaml"
    dataset_path.write_text(yaml.safe_dump(dataset_manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")

    split_rows = [
        {"subject_id": row["subject_id"], "original_trial_id": f"{row['subject_id']}::{row['trial_id']}", "split": row["split"]}
        for row in trial_index
    ]
    split_manifest = {
        "split_id": active_split_id,
        "subject_group_split": True,
        "status": "ready",
        "description": f"Subject holdout split for {active_split_id}, {len(train_subjects)} train / {len(val_subjects)} val / {len(test_subjects)} test subjects.",
        "train_subjects": sorted(train_subjects),
        "val_subjects": sorted(val_subjects),
        "test_subjects": sorted(test_subjects),
        "trial_rows": split_rows,
    }
    split_path = run_dir / "split_manifest.yaml"
    split_path.write_text(yaml.safe_dump(split_manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return dataset_path, split_path


def _resolve_device(device_str: str = "auto") -> str:
    if device_str == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device_str


@dataclass(frozen=True)
class RealRunArtifacts:
    run_dir: Path
    dataset_manifest: Path
    split_manifest: Path
    prediction_csv: Path
    score_matrix_csv: Path
    manifest_json: Path
    metric_report: Mapping[str, Any]


VALID_RUN_MODES = frozenset({"smoke", "full_subjects", "candidate"})
"""Known run modes.

``smoke`` uses a seeded subset and val-only predictions.
``full_subjects`` uses all subjects but remains a val-only diagnostic run.
``candidate`` uses all subjects and writes held-out test predictions.
"""

VALID_PREDICTION_SCOPES = frozenset({"val_only", "test_only"})
"""Known prediction scopes."""


def run_real_classifier_route(
    *,
    route_config_path: Path,
    run_dir: Path,
    run_mode: str = "smoke",
    split_id: str | None = None,
    seed: int | None = None,
    command: str | Sequence[str] = "python scripts/train_route.py",
    data_root: Path | None = None,
    smoke_n_dep: int = 4,
    smoke_n_hc: int = 8,
    smoke_epochs: int | None = None,
    device: str = "auto",
) -> RealRunArtifacts:
    """Run a real EEG classifier route end-to-end.

    Args:
        route_config_path: Path to route YAML config.
        run_dir: Output directory for all artifacts.
        run_mode: "smoke" (subset + val-only), "full_subjects" (all subjects + val-only),
            or "candidate" (all subjects + held-out test predictions).
        split_id: Override split_id (default from route config).
        seed: Override seed (default from route config).
        command: Command string for manifest provenance.
        data_root: Path to .mat data root directory.
        smoke_n_dep: Number of DEP subjects to use (smoke mode).
        smoke_n_hc: Number of HC subjects to use (smoke mode).
        smoke_epochs: Override epoch count (smoke mode).
        device: Torch device.
    """
    if run_mode not in VALID_RUN_MODES:
        raise ValueError(f"unknown run_mode: {run_mode}; valid: {', '.join(sorted(VALID_RUN_MODES))}")
    import torch
    from torch.utils.data import DataLoader
    from hust_bci_er.models.factory import build_model
    from hust_bci_er.training.classifier import (
        ClassifierTrainConfig,
        EarlyStoppingConfig,
        OptimizerConfig,
        TrainResult,
        fit_classifier,
    )

    route_config_path = route_config_path.resolve()
    route_data = yaml.safe_load(route_config_path.read_text(encoding="utf-8")) or {}
    errors = validate_route_config(route_data, path=route_config_path)
    if errors:
        raise ValueError(f"invalid route config: {errors}")

    route_id = str(route_data["route_id"])
    active_seed = int(seed if seed is not None else route_data["seed"])
    active_split_id = split_id or str(route_data["split_id"])

    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load and subset data
    data_root = _resolve_data_root(data_root)
    all_trials = _load_mat_trials(data_root)
    if not all_trials:
        raise FileNotFoundError(f"no .mat files found under {data_root}")

    rng = np.random.default_rng(active_seed)
    dep_subjects = sorted(set(t["subject_id"] for t in all_trials if t["cohort"] == "DEP"))
    hc_subjects = sorted(set(t["subject_id"] for t in all_trials if t["cohort"] == "HC"))

    if run_mode in {"full_subjects", "candidate"}:
        # Use all subjects with 20% val + 20% test per cohort
        n_dep = len(dep_subjects)
        n_hc = len(hc_subjects)
        val_dep = max(1, n_dep // 5)
        val_hc = max(1, n_hc // 5)
        trials = all_trials
    else:
        # Smoke: subset subjects by seeded random selection
        n_dep_avail = len(dep_subjects)
        n_hc_avail = len(hc_subjects)
        subset_dep = list(rng.choice(dep_subjects, min(smoke_n_dep, n_dep_avail), replace=False))
        subset_hc = list(rng.choice(hc_subjects, min(smoke_n_hc, n_hc_avail), replace=False))
        subset_ids = set(subset_dep) | set(subset_hc)
        trials = [t for t in all_trials if t["subject_id"] in subset_ids]
        val_dep = max(1, smoke_n_dep // 2)
        val_hc = max(1, smoke_n_hc // 2)

    # Split subjects
    train_subjects, val_subjects, test_subjects = _split_subjects(
        trials, val_dep=val_dep, val_hc=val_hc, seed=active_seed
    )

    train_trials = [t for t in trials if t["subject_id"] in train_subjects]
    val_trials = [t for t in trials if t["subject_id"] in val_subjects]
    test_trials = [t for t in trials if t["subject_id"] in test_subjects]

    # Determine window configuration
    augmentation = route_data.get("augmentation")
    aug_name = augmentation.get("name") if isinstance(augmentation, dict) else None
    has_sliding_aug = aug_name == "split_first_sliding_window"
    has_fixed_crop_aug = aug_name == "split_first_fixed_crops"
    has_aug = has_sliding_aug or has_fixed_crop_aug
    inference = route_data.get("inference") if isinstance(route_data, dict) else None
    crop_policy = str(inference.get("crop_policy", "single")) if isinstance(inference, dict) else "single"
    input_window_sec = float(route_data.get("input_window_sec", 10))
    preproc = list(route_data.get("preprocessing", []) or [])
    _validate_adapter_preprocessing(preproc)

    if has_sliding_aug:
        source_trial_sec = float(augmentation["source_trial_sec"])
        window_sec = float(augmentation["window_sec"])
        stride_sec = float(augmentation["stride_sec"])
        n_fixed_crops = None
    elif has_fixed_crop_aug:
        source_trial_sec = float(augmentation["source_trial_sec"])
        window_sec = float(augmentation["window_sec"])
        stride_sec = window_sec
        n_fixed_crops = int(augmentation["n_crops"])
    else:
        source_trial_sec = input_window_sec
        window_sec = input_window_sec
        stride_sec = input_window_sec
        n_fixed_crops = None

    if has_fixed_crop_aug:
        _validate_fixed_crop_coverage(
            trials,
            source_trial_sec=source_trial_sec,
            window_sec=window_sec,
            n_crops=int(n_fixed_crops),
            split_name=run_mode,
        )
    elif run_mode == "candidate" and not has_aug:
        _validate_fixed_crop_coverage(
            trials,
            source_trial_sec=input_window_sec * FIXED_CANDIDATE_CROPS,
            window_sec=input_window_sec,
            n_crops=FIXED_CANDIDATE_CROPS,
            split_name="candidate",
        )

    # Create raw windows first (skip preprocessing), fit EA on all training
    # windows if needed, then apply preprocessing to everything.
    if has_sliding_aug:
        make_windows = _make_sliding_windows
        window_kwargs: dict = dict(source_trial_sec=source_trial_sec, window_sec=window_sec, stride_sec=stride_sec)
    elif has_fixed_crop_aug:
        make_windows = _make_fixed_crops
        window_kwargs = dict(source_trial_sec=source_trial_sec, window_sec=window_sec, n_crops=int(n_fixed_crops))
    else:
        make_windows = _make_single_crops
        window_kwargs = dict(window_sec=window_sec)
    raw_train = make_windows(train_trials, preproc=preproc, skip_preproc=True, **window_kwargs)
    raw_val = make_windows(val_trials, preproc=preproc, skip_preproc=True, **window_kwargs)
    if run_mode == "candidate":
        if not test_trials:
            raise ValueError("candidate mode requires non-empty test subjects")
        if has_aug:
            raw_eval = make_windows(test_trials, preproc=preproc, skip_preproc=True, **window_kwargs)
        else:
            raw_eval = _make_fixed_crops(
                test_trials,
                source_trial_sec=input_window_sec * FIXED_CANDIDATE_CROPS,
                window_sec=input_window_sec,
                n_crops=FIXED_CANDIDATE_CROPS,
                preproc=preproc,
                skip_preproc=True,
            )
        eval_split = "test"
    else:
        raw_eval = raw_val
        eval_split = "val"

    if not raw_train:
        raise ValueError("no training windows created")
    if not raw_val:
        raise ValueError("no validation windows created")
    if not raw_eval:
        raise ValueError(f"no {eval_split} windows created")

    # Fit EA on all raw training windows
    ea_transform = _fit_ea_on_windows(raw_train, preproc)

    # Apply preprocessing (including EA if fitted)
    def _apply_preproc_to_windows(windows, ea):
        for w in windows:
            w["x"] = _apply_preprocessing(w["x"], preproc, ea_transform=ea)

    _apply_preproc_to_windows(raw_train, ea_transform)
    _apply_preproc_to_windows(raw_val, ea_transform)
    if raw_eval is not raw_val:
        _apply_preproc_to_windows(raw_eval, ea_transform)
    train_windows = raw_train
    val_windows = raw_val
    eval_windows = raw_eval
    train_transform_configs = transform_configs_from_route(route_data)
    if train_transform_configs:
        train_windows = apply_transforms_to_windows(
            train_windows,
            train_transform_configs,
            seed=active_seed,
            split="train",
        )

    # Build model
    model_config = route_data.get("model") or {}
    model_name = str(model_config.get("name", "")) if isinstance(model_config, dict) else str(model_config)
    if not model_name:
        raise ValueError("route config must specify model.name")
    model_kwargs = (
        {key: value for key, value in model_config.items() if key != "name"}
        if isinstance(model_config, dict)
        else {}
    )
    n_times = int(round(window_sec * SFREQ))
    model = build_model(
        model_name,
        n_channels=30,
        n_times=n_times,
        n_classes=2,
        **model_kwargs,
    )

    # Build training config
    training_config = route_data.get("training")
    source_epochs = int(training_config["epochs"]) if isinstance(training_config, dict) else 80
    epochs = smoke_epochs if smoke_epochs is not None else source_epochs
    batch_size = int(training_config["batch_size"]) if isinstance(training_config, dict) else 32

    opt_data = training_config.get("optimizer") if isinstance(training_config, dict) else None
    optimizer = OptimizerConfig(
        name=str(opt_data["name"]) if isinstance(opt_data, dict) else "adamw",
        lr=float(opt_data["lr"]) if isinstance(opt_data, dict) else 1e-3,
        weight_decay=float(opt_data.get("weight_decay", 0.01)) if isinstance(opt_data, dict) else 0.01,
    )

    early_data = training_config.get("early_stopping") if isinstance(training_config, dict) else None
    early_stopping = EarlyStoppingConfig(
        monitor=str(early_data.get("monitor", "val_loss")),
        mode=str(early_data.get("mode", "min")),
        patience=int(early_data.get("patience", 12)),
    ) if isinstance(early_data, dict) else None

    grad_clip = float(training_config["grad_clip_norm"]) if isinstance(training_config, dict) and "grad_clip_norm" in training_config else None

    device_str = _resolve_device(device)
    train_config = ClassifierTrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        device=device_str,
        seed=active_seed,
        optimizer=optimizer,
        early_stopping=early_stopping,
        grad_clip_norm=grad_clip,
    )

    # Train
    train_loader = DataLoader(_WindowDataset(train_windows), batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataset = _WindowDataset(val_windows)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    result: TrainResult = fit_classifier(model, train_loader, val_loader=val_loader, config=train_config)

    # Predict on the audited split. Smoke/full_subjects are val-only diagnostics;
    # candidate mode evaluates held-out test subjects.
    eval_window_rows = _predict_scores(model, eval_windows, batch_size=batch_size, device=device_str)

    # Aggregate to trial level for predictions.csv
    aug_config = route_data.get("augmentation")
    if isinstance(aug_config, dict) and aug_config.get("aggregate_to_trial"):
        agg = aug_config["aggregate_to_trial"]
        method = str(agg.get("method", "majority_vote"))
        tie_break = str(agg.get("tie_break", "mean_score"))
    else:
        method = "mean_score"
        tie_break = "mean_score"
    eval_trial_rows = _aggregate_to_trials(eval_window_rows, method=method, tie_break=tie_break)

    # Build score matrix (5 crops per trial) from window rows
    score_matrix_rows, score_matrix_evidence = _build_score_matrix(eval_window_rows, crop_policy=crop_policy, seed=active_seed)
    score_matrix_path = run_dir / "score_matrix.csv"
    _write_score_matrix(score_matrix_path, score_matrix_rows)

    # Build prediction records from trial-level rows
    prediction_records = [
        PredictionRecord(
            route_id=route_id,
            subject_id=str(row["subject_id"]),
            trial_id=str(row["trial_id"]),
            crop_id=0,
            y_score=float(row["y_score"]),
            y_pred=int(row["y_pred"]),
            y_true=int(row["y_true"]),
            seed=active_seed,
        )
        for row in eval_trial_rows
    ]
    prediction_path = run_dir / "predictions.csv"
    write_predictions(prediction_records, prediction_path, group_keys=("subject_id",))

    # Build metric report
    primary_metric = str(route_data["evaluation"]["primary_metric"])
    metric_report = build_metric_report(
        route_id=route_id,
        prediction_csv=prediction_path,
        score_matrix_csv=score_matrix_path,
        primary_metric=primary_metric,
    )
    write_metric_report(metric_report, run_dir)

    # Write evidence manifests
    prediction_scope = "test_only" if run_mode == "candidate" else "val_only"

    dataset_path, split_path = _write_evidence_manifests(
        run_dir,
        route_data=route_data,
        active_split_id=active_split_id,
        all_trials=trials,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        data_root=data_root,
        run_mode=run_mode,
    )

    # Write model state
    (run_dir / "model_state.local.json").write_text(
        json.dumps({
            "adapter": "torch_classifier",
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "best_epoch": result.best_epoch,
            "best_metric": result.best_metric,
            "epochs_ran": len(result.history),
            "training_epochs": epochs,
            "source_training_epochs": source_epochs,
            "training_epochs_overridden": epochs != source_epochs,
            "seed": active_seed,
            "run_mode": run_mode,
            "prediction_scope": prediction_scope,
            "evaluation_split": eval_split,
            "score_matrix_evidence": score_matrix_evidence,
            "augmentation_transforms": train_transform_configs,
            "augmentation_transform_scope": "train_only" if train_transform_configs else "none",
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Write manifest.json
    context = prepare_run_manifest_context(
        route_config_path=route_config_path,
        run_dir=run_dir,
        prediction_csv=prediction_path,
        dataset_manifest_path=dataset_path,
        split_manifest_path=split_path,
        split_id=active_split_id,
        source_split_id=str(route_data["split_id"]),
        seed=active_seed,
        source_seed=int(route_data["seed"]),
    )
    manifest = build_run_manifest_payload(
        context,
        metrics=metric_report["metrics"],
        command=command,
        score_matrix_csv=score_matrix_path,
    )
    manifest["run_mode"] = run_mode
    manifest["model_name"] = model_name
    manifest["model_kwargs"] = model_kwargs
    manifest["prediction_scope"] = prediction_scope
    manifest["evaluation_split"] = eval_split
    manifest["evaluation_subjects"] = sorted(test_subjects if eval_split == "test" else val_subjects)
    manifest["split_subject_counts"] = {
        "train": len(train_subjects),
        "val": len(val_subjects),
        "test": len(test_subjects),
    }
    manifest["training_epochs"] = epochs
    manifest["source_training_epochs"] = source_epochs
    manifest["training_epochs_overridden"] = epochs != source_epochs
    manifest["score_matrix_evidence"] = score_matrix_evidence
    manifest["augmentation_transforms"] = train_transform_configs
    manifest["augmentation_transform_scope"] = "train_only" if train_transform_configs else "none"
    manifest["declared_protocol"] = str(route_data["evaluation"]["protocol"])
    manifest["adapter_execution_protocol"] = "single_subject_holdout_split"
    manifest["protocol_evidence_scope"] = (
        "one materialized split/fold from the declared route protocol; "
        "full repeated/nested protocol evidence must be produced by run_evaluation_protocol.py"
    )
    if run_mode in {"full_subjects", "candidate"}:
        ds = yaml.safe_load(dataset_path.read_text(encoding="utf-8")) or {}
        manifest["raw_data_sources"] = ds.get("raw_data_sources") or []
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return RealRunArtifacts(
        run_dir=run_dir,
        dataset_manifest=dataset_path,
        split_manifest=split_path,
        prediction_csv=prediction_path,
        score_matrix_csv=score_matrix_path,
        manifest_json=run_dir / "manifest.json",
        metric_report=metric_report,
    )


def _write_score_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["subject_id", "trial_id", "y_true"]
    for idx in range(5):
        fieldnames.extend([f"crop_{idx}", f"crop_{idx}_source_crop_id", f"crop_{idx}_window_start_sec"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
