"""Real EEG classifier job adapter for .mat dataset routes.

Wires route config -> .mat loading -> preprocessing -> augmentation -> torch
training -> inference -> prediction & score_matrix artifacts -> audit manifests.
"""

from __future__ import annotations

import csv
import json
import os
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
from hust_bci_er.evaluation.crop_policy import FIXED_CROP_POLICIES
from hust_bci_er.evaluation.exact_single_crop import assignment_grid, top4_predictions
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
TORCH_NUM_THREADS_ENV = "HUST_TORCH_NUM_THREADS"
TORCH_INTEROP_THREADS_ENV = "HUST_TORCH_INTEROP_THREADS"
DATALOADER_NUM_WORKERS_ENV = "HUST_DATALOADER_NUM_WORKERS"
DATALOADER_PIN_MEMORY_ENV = "HUST_DATALOADER_PIN_MEMORY"
DATALOADER_PERSISTENT_WORKERS_ENV = "HUST_DATALOADER_PERSISTENT_WORKERS"
DATALOADER_PREFETCH_FACTOR_ENV = "HUST_DATALOADER_PREFETCH_FACTOR"

_TORCH_INTEROP_THREADS_CONFIGURED: int | None = None


def _env_int(name: str, *, minimum: int) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def _env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0/true/false/yes/no/on/off, got {value!r}")


def _configure_torch_runtime(torch_module) -> dict[str, Any]:
    """Apply opt-in torch runtime caps for shared training hosts."""
    global _TORCH_INTEROP_THREADS_CONFIGURED

    applied: dict[str, Any] = {
        "num_threads_env": None,
        "interop_threads_env": None,
        "num_threads_applied": None,
        "interop_threads_applied": None,
    }
    num_threads = _env_int(TORCH_NUM_THREADS_ENV, minimum=1)
    if num_threads is not None:
        torch_module.set_num_threads(num_threads)
        applied["num_threads_env"] = TORCH_NUM_THREADS_ENV
        applied["num_threads_applied"] = num_threads

    interop_threads = _env_int(TORCH_INTEROP_THREADS_ENV, minimum=1)
    if interop_threads is not None:
        applied["interop_threads_env"] = TORCH_INTEROP_THREADS_ENV
        if _TORCH_INTEROP_THREADS_CONFIGURED is None:
            torch_module.set_num_interop_threads(interop_threads)
            _TORCH_INTEROP_THREADS_CONFIGURED = interop_threads
        elif _TORCH_INTEROP_THREADS_CONFIGURED != interop_threads:
            raise ValueError(
                f"{TORCH_INTEROP_THREADS_ENV} changed from "
                f"{_TORCH_INTEROP_THREADS_CONFIGURED} to {interop_threads} in one process"
            )
        applied["interop_threads_applied"] = _TORCH_INTEROP_THREADS_CONFIGURED
    return applied


def _dataloader_kwargs(*, device: str) -> dict[str, Any]:
    """Build DataLoader kwargs while preserving serial defaults unless env opts in."""
    num_workers = _env_int(DATALOADER_NUM_WORKERS_ENV, minimum=0)
    if num_workers is None:
        num_workers = 0

    pin_memory = _env_bool(DATALOADER_PIN_MEMORY_ENV)
    if pin_memory is None:
        pin_memory = bool(num_workers > 0 and device.startswith("cuda"))

    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        persistent_workers = _env_bool(DATALOADER_PERSISTENT_WORKERS_ENV)
        kwargs["persistent_workers"] = True if persistent_workers is None else persistent_workers
        prefetch_factor = _env_int(DATALOADER_PREFETCH_FACTOR_ENV, minimum=1)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


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


def _subject_sets_from_split_manifest(path: Path, *, active_split_id: str) -> tuple[set[str], set[str], set[str]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"split manifest must be a mapping: {path}")
    manifest_split_id = str(data.get("split_id") or "")
    if manifest_split_id and manifest_split_id != active_split_id:
        raise ValueError(f"split manifest split_id {manifest_split_id} does not match active split_id {active_split_id}")

    def listed(name: str) -> set[str]:
        value = data.get(name) or []
        if not isinstance(value, list):
            raise ValueError(f"split manifest {name} must be a list")
        return {str(item) for item in value}

    train_subjects = listed("train_subjects")
    val_subjects = listed("val_subjects")
    test_subjects = listed("test_subjects")
    if not (train_subjects or val_subjects or test_subjects):
        rows = data.get("trial_rows") or []
        if not isinstance(rows, list) or not rows:
            raise ValueError("split manifest must contain subject lists or trial_rows")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("split manifest trial_rows must contain only mappings")
            subject = str(row.get("subject_id") or "")
            split = str(row.get("split") or "")
            if not subject:
                raise ValueError("split manifest trial row is missing subject_id")
            if split == "train":
                train_subjects.add(subject)
            elif split == "val":
                val_subjects.add(subject)
            elif split == "test":
                test_subjects.add(subject)
            else:
                raise ValueError(f"split manifest trial row has invalid split: {split}")
    overlaps = {
        "train_val": train_subjects & val_subjects,
        "train_test": train_subjects & test_subjects,
        "val_test": val_subjects & test_subjects,
    }
    bad = {key: sorted(value) for key, value in overlaps.items() if value}
    if bad:
        raise ValueError(f"split manifest subject overlap: {bad}")
    if not train_subjects:
        raise ValueError("split manifest must provide non-empty train_subjects")
    if not val_subjects:
        raise ValueError("split manifest must provide non-empty val_subjects for checkpoint selection")
    if not test_subjects:
        raise ValueError("split manifest must provide non-empty test_subjects")
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

    _configure_torch_runtime(torch)
    loader = DataLoader(
        _WindowDataset(windows),
        batch_size=batch_size,
        shuffle=False,
        **_dataloader_kwargs(device=device),
    )
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
    crop_policy: str | Mapping[str, Any],
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
    policy_name = str(crop_policy.get("name") if isinstance(crop_policy, Mapping) else crop_policy)
    if policy_name in {*FIXED_CROP_POLICIES, "random", "worst"}:
        return _apply_single_crop_policy_to_score_matrix(rows, crop_policy=crop_policy, seed=seed), "genuine" if all_genuine else "synthetic"
    return rows, "genuine" if all_genuine else "synthetic"


def _apply_single_crop_policy_to_score_matrix(
    rows: list[dict[str, Any]],
    *,
    crop_policy: str | Mapping[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    """Collapse five-crop matrices to a declared P2 single-crop policy.

    The output intentionally remains a 5-column score matrix by repeating the
    selected crop.  That lets existing exact-metric code evaluate the selected
    single-crop policy without pretending the repeated columns are genuine
    five-crop evidence.
    """
    policy_name = str(crop_policy.get("name") if isinstance(crop_policy, Mapping) else crop_policy)
    rng_seed = int(crop_policy.get("random_seed", seed)) if isinstance(crop_policy, Mapping) else int(seed)
    rng = np.random.default_rng(rng_seed)
    selected_by_key: dict[tuple[str, str], int] = {}
    if policy_name == "worst":
        by_subject: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_subject.setdefault(str(row["subject_id"]), []).append(row)
        for subject_rows in by_subject.values():
            subject_rows = sorted(subject_rows, key=lambda row: str(row["trial_id"]))
            if len(subject_rows) != 8:
                for row in subject_rows:
                    crop_values = [float(row[f"crop_{idx}"]) for idx in range(5)]
                    selected_by_key[(str(row["subject_id"]), str(row["trial_id"]))] = int(np.argmin(crop_values))
                continue
            mat = np.array([[float(row[f"crop_{idx}"]) for idx in range(5)] for row in subject_rows], dtype=np.float64)
            y_true = np.array([int(float(row["y_true"])) for row in subject_rows], dtype=np.int8)
            grid = assignment_grid(mat.shape[0], mat.shape[1])
            selected = mat[np.arange(mat.shape[0]), grid]
            pred = top4_predictions(selected)
            values = (pred == y_true[None, :]).mean(axis=1)
            assignment = grid[int(np.argmin(values))]
            for row, crop_idx in zip(subject_rows, assignment):
                selected_by_key[(str(row["subject_id"]), str(row["trial_id"]))] = int(crop_idx)

    out: list[dict[str, Any]] = []
    for row in rows:
        if policy_name in FIXED_CROP_POLICIES:
            selected_idx = int(FIXED_CROP_POLICIES[policy_name])
        elif policy_name == "random":
            selected_idx = int(rng.integers(0, 5))
        elif policy_name == "worst":
            selected_idx = selected_by_key[(str(row["subject_id"]), str(row["trial_id"]))]
        else:
            selected_idx = 0
        selected_score = row[f"crop_{selected_idx}"]
        selected_source_crop = row.get(f"crop_{selected_idx}_source_crop_id", selected_idx)
        selected_start = row.get(f"crop_{selected_idx}_window_start_sec", float(selected_idx))
        item = dict(row)
        for idx in range(5):
            item[f"crop_{idx}"] = selected_score
            item[f"crop_{idx}_source_crop_id"] = selected_source_crop
            item[f"crop_{idx}_window_start_sec"] = selected_start
        out.append(item)
    return out


def _is_protocol_single_crop_policy(policy: str | Mapping[str, Any]) -> bool:
    name = str(policy.get("name") if isinstance(policy, Mapping) else policy)
    return name in {*FIXED_CROP_POLICIES, "random", "worst"}


def _trial_rows_from_score_matrix(score_matrix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in score_matrix_rows:
        scores = [float(row[f"crop_{idx}"]) for idx in range(5)]
        score = float(sum(scores) / len(scores))
        rows.append(
            {
                "subject_id": row["subject_id"],
                "trial_id": row["trial_id"],
                "y_true": int(float(row["y_true"])),
                "y_score": score,
                "y_pred": int(score >= 0.5),
            }
        )
    return rows


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


def _torch_load_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    # weights_only=False is required to load full protocol checkpoints that
    # include numpy arrays (ea_transform) and arbitrary Python objects.
    # These checkpoints are only loaded from paths the protocol runner itself
    # wrote; never pass user-supplied paths to this function.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint must contain a mapping payload: {path}")
    return payload


def _checkpoint_ea_transform(payload: Mapping[str, Any], preproc: Sequence[Any]) -> np.ndarray | None:
    raw = payload.get("ea_transform")
    if raw is None:
        if "euclidean_alignment" in _preprocessing_names(preproc):
            raise ValueError("reused checkpoint is missing euclidean_alignment transform")
        return None
    return np.asarray(raw, dtype=np.float32)


def _validate_reuse_checkpoint(
    payload: Mapping[str, Any],
    *,
    route_id: str,
    split_id: str,
    seed: int,
    model_name: str,
    model_kwargs: Mapping[str, Any],
    n_times: int,
    preproc: Sequence[Any],
    run_mode: str,
    reuse_checkpoint_context: Mapping[str, Any] | None = None,
) -> None:
    if payload.get("artifact_kind") != "torch_classifier_checkpoint":
        raise ValueError("reuse checkpoint is not a torch_classifier_checkpoint artifact")
    p3_selected_checkpoint = (
        isinstance(reuse_checkpoint_context, Mapping)
        and str(reuse_checkpoint_context.get("protocol", "")) == "p3_nested_selection"
        and str(reuse_checkpoint_context.get("stage", "")) == "inner_select"
    )
    mismatches: list[str] = []
    if str(payload.get("route_id", "")) != route_id:
        mismatches.append("route_id")
    if str(payload.get("split_id", "")) != split_id and not p3_selected_checkpoint:
        mismatches.append("split_id")
    if int(payload.get("seed", -1)) != int(seed) and not p3_selected_checkpoint:
        mismatches.append("seed")
    if p3_selected_checkpoint:
        if str(payload.get("job_id", "")) != str(reuse_checkpoint_context.get("source_job_id", "")):
            mismatches.append("source_job_id")
        if payload.get("outer_fold") != reuse_checkpoint_context.get("outer_fold"):
            mismatches.append("outer_fold")
        if payload.get("param_index") != reuse_checkpoint_context.get("selected_param_index"):
            mismatches.append("param_index")
    if str(payload.get("model_name", "")) != model_name:
        mismatches.append("model_name")
    if dict(payload.get("model_kwargs") or {}) != dict(model_kwargs or {}):
        mismatches.append("model_kwargs")
    if int(payload.get("n_times", -1)) != int(n_times):
        mismatches.append("n_times")
    if list(payload.get("preprocessing") or []) != list(preproc):
        mismatches.append("preprocessing")
    if mismatches:
        raise ValueError("reuse checkpoint does not match this protocol job: " + ", ".join(mismatches))
    if run_mode == "candidate" and bool(payload.get("training_epochs_overridden")):
        raise ValueError("candidate evaluation cannot reuse a checkpoint trained with an epoch override")
    if not isinstance(payload.get("model_state_dict"), Mapping):
        raise ValueError("reuse checkpoint is missing model_state_dict")


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
    split_manifest_path: Path | None = None,
    seed: int | None = None,
    command: str | Sequence[str] = "python scripts/train_route.py",
    data_root: Path | None = None,
    smoke_n_dep: int = 4,
    smoke_n_hc: int = 8,
    smoke_epochs: int | None = None,
    device: str = "auto",
    crop_policy: Mapping[str, Any] | None = None,
    save_checkpoint_path: Path | None = None,
    reuse_checkpoint_path: Path | None = None,
    reuse_checkpoint_context: Mapping[str, Any] | None = None,
) -> RealRunArtifacts:
    """Run a real EEG classifier route end-to-end.

    Args:
        route_config_path: Path to route YAML config.
        run_dir: Output directory for all artifacts.
        run_mode: "smoke" (subset + val-only), "full_subjects" (all subjects + val-only),
            or "candidate" (all subjects + held-out test predictions).
        split_id: Override split_id (default from route config).
        split_manifest_path: Optional formal split manifest with train/val/test subjects.
        seed: Override seed (default from route config).
        command: Command string for manifest provenance.
        data_root: Path to .mat data root directory.
        smoke_n_dep: Number of DEP subjects to use (smoke mode).
        smoke_n_hc: Number of HC subjects to use (smoke mode).
        smoke_epochs: Override epoch count (smoke mode).
        device: Torch device.
        crop_policy: Optional protocol job crop-policy override, used by P2 diagnostics.
        save_checkpoint_path: Optional path for a reusable torch classifier checkpoint.
        reuse_checkpoint_path: Optional reusable checkpoint path for evaluation-only jobs.
        reuse_checkpoint_context: Optional protocol context for validating reusable checkpoint provenance.
    """
    if run_mode not in VALID_RUN_MODES:
        raise ValueError(f"unknown run_mode: {run_mode}; valid: {', '.join(sorted(VALID_RUN_MODES))}")
    import torch
    from torch.utils.data import DataLoader
    from hust_bci_er.models.factory import build_model
    from hust_bci_er.training.classifier import (
        ClassifierTrainConfig,
        EarlyStoppingConfig,
        EpochMetrics,
        OptimizerConfig,
        TrainResult,
        fit_classifier,
        set_torch_seed,
    )
    from hust_bci_er.training.monitor import TrainingMonitor

    torch_runtime_config = _configure_torch_runtime(torch)

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
    monitor = TrainingMonitor(run_dir, route_id=route_id, run_id=run_dir.name)
    monitor.log(
        "start",
        {
            "run_mode": run_mode,
            "split_id": active_split_id,
            "seed": active_seed,
            "device": device,
            "resume_checkpoint": "training_checkpoint.pt",
        },
    )

    # Load and subset data
    data_root = _resolve_data_root(data_root)
    all_trials = _load_mat_trials(data_root)
    if not all_trials:
        raise FileNotFoundError(f"no .mat files found under {data_root}")

    dep_subjects = sorted(set(t["subject_id"] for t in all_trials if t["cohort"] == "DEP"))
    hc_subjects = sorted(set(t["subject_id"] for t in all_trials if t["cohort"] == "HC"))

    if split_manifest_path is not None:
        train_subjects, val_subjects, test_subjects = _subject_sets_from_split_manifest(
            Path(split_manifest_path),
            active_split_id=active_split_id,
        )
        available_subjects = {t["subject_id"] for t in all_trials}
        requested_subjects = train_subjects | val_subjects | test_subjects
        missing_subjects = sorted(requested_subjects - available_subjects)
        if missing_subjects:
            raise ValueError(f"split manifest references subjects absent from data_root: {missing_subjects[:5]}")
        trials = [t for t in all_trials if t["subject_id"] in requested_subjects]
    elif run_mode in {"full_subjects", "candidate"}:
        # Use all subjects with 20% val + 20% test per cohort
        n_dep = len(dep_subjects)
        n_hc = len(hc_subjects)
        val_dep = max(1, n_dep // 5)
        val_hc = max(1, n_hc // 5)
        trials = all_trials
        train_subjects, val_subjects, test_subjects = _split_subjects(
            trials, val_dep=val_dep, val_hc=val_hc, seed=active_seed
        )
    else:
        # Smoke: subset subjects by seeded random selection
        rng = np.random.default_rng(active_seed)
        n_dep_avail = len(dep_subjects)
        n_hc_avail = len(hc_subjects)
        subset_dep = list(rng.choice(dep_subjects, min(smoke_n_dep, n_dep_avail), replace=False))
        subset_hc = list(rng.choice(hc_subjects, min(smoke_n_hc, n_hc_avail), replace=False))
        subset_ids = set(subset_dep) | set(subset_hc)
        trials = [t for t in all_trials if t["subject_id"] in subset_ids]
        val_dep = max(1, smoke_n_dep // 2)
        val_hc = max(1, smoke_n_hc // 2)
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
    route_crop_policy = str(inference.get("crop_policy", "single")) if isinstance(inference, dict) else "single"
    active_crop_policy: Mapping[str, Any] | str = dict(crop_policy) if isinstance(crop_policy, Mapping) else route_crop_policy
    input_window_sec = float(route_data.get("input_window_sec", 10))
    preproc = list(route_data.get("preprocessing", []) or [])
    _validate_adapter_preprocessing(preproc)
    checkpoint_payload: Mapping[str, Any] | None = None
    checkpoint_reuse: dict[str, Any] | None = None
    checkpoint_ea_transform: np.ndarray | None = None
    if reuse_checkpoint_path is not None:
        reuse_checkpoint_path = Path(reuse_checkpoint_path).resolve()
        checkpoint_payload = _torch_load_checkpoint(reuse_checkpoint_path)
        checkpoint_ea_transform = _checkpoint_ea_transform(checkpoint_payload, preproc)

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

    # Fit EA on all raw training windows, or reuse the transform bound to the
    # checkpoint that produced this protocol model.
    ea_transform = checkpoint_ea_transform if checkpoint_payload is not None else _fit_ea_on_windows(raw_train, preproc)

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
    set_torch_seed(active_seed)
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
    reset_parameters_after_seed = (
        bool(training_config.get("reset_parameters_after_seed", True))
        if isinstance(training_config, dict)
        else True
    )

    device_str = _resolve_device(device)
    dataloader_config = _dataloader_kwargs(device=device_str)
    train_config = ClassifierTrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        device=device_str,
        seed=active_seed,
        optimizer=optimizer,
        early_stopping=early_stopping,
        grad_clip_norm=grad_clip,
        reset_parameters_after_seed=reset_parameters_after_seed,
    )

    result: TrainResult | None
    if checkpoint_payload is not None:
        _validate_reuse_checkpoint(
            checkpoint_payload,
            route_id=route_id,
            split_id=active_split_id,
            seed=active_seed,
            model_name=model_name,
            model_kwargs=model_kwargs,
            n_times=n_times,
            preproc=preproc,
            run_mode=run_mode,
            reuse_checkpoint_context=reuse_checkpoint_context,
        )
        model.load_state_dict(checkpoint_payload["model_state_dict"])
        model.to(device_str)
        checkpoint_reuse = {
            "checkpoint_path": str(Path(reuse_checkpoint_path).resolve()),
            "checkpoint_sha256": sha256_file(Path(reuse_checkpoint_path)),
            "source_job_id": checkpoint_payload.get("job_id"),
            "source_run_dir": checkpoint_payload.get("run_dir"),
            "checkpoint_schema_version": checkpoint_payload.get("checkpoint_schema_version"),
        }
        if reuse_checkpoint_context is not None:
            checkpoint_reuse["selection_context"] = dict(reuse_checkpoint_context)
        result = None
    else:
        # Train
        train_loader = DataLoader(
            _WindowDataset(train_windows),
            batch_size=batch_size,
            shuffle=True,
            **dataloader_config,
        )
        val_dataset = _WindowDataset(val_windows)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **dataloader_config)

        def _log_epoch(metrics: EpochMetrics) -> None:
            monitor.epoch(
                metrics.epoch,
                {
                    "train_loss": metrics.train_loss,
                    "train_accuracy": metrics.train_accuracy,
                    "n_train": metrics.n_train,
                    "val_loss": metrics.val_loss,
                    "val_accuracy": metrics.val_accuracy,
                    "n_val": metrics.n_val,
                },
            )

        training_resume_checkpoint_path = run_dir / "training_checkpoint.pt"
        result = fit_classifier(
            model,
            train_loader,
            val_loader=val_loader,
            config=train_config,
            checkpoint_path=training_resume_checkpoint_path,
            resume=True,
            epoch_callback=_log_epoch,
        )
        if save_checkpoint_path is not None:
            save_checkpoint_path = Path(save_checkpoint_path).resolve()
            save_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            if result.best_state_dict is None:
                raise ValueError("cannot save reusable checkpoint without best_state_dict")
            torch.save(
                {
                    "checkpoint_schema_version": 1,
                    "artifact_kind": "torch_classifier_checkpoint",
                    "adapter": "torch_classifier",
                    "job_id": None,
                    "route_id": route_id,
                    "split_id": active_split_id,
                    "source_split_id": str(route_data["split_id"]),
                    "seed": active_seed,
                    "source_seed": int(route_data["seed"]),
                    "model_name": model_name,
                    "model_kwargs": model_kwargs,
                    "n_times": n_times,
                    "window_sec": window_sec,
                    "input_window_sec": input_window_sec,
                    "preprocessing": preproc,
                    "augmentation": augmentation if isinstance(augmentation, Mapping) else None,
                    "ea_transform": ea_transform.tolist() if ea_transform is not None else None,
                    "model_state_dict": result.best_state_dict,
                    "best_epoch": int(result.best_epoch),
                    "best_metric": float(result.best_metric),
                    "epochs_ran": len(result.history),
                    "training_epochs": int(epochs),
                    "source_training_epochs": int(source_epochs),
                    "training_epochs_overridden": bool(epochs != source_epochs),
                    "checkpoint_selection": {
                        "rule": "best_monitored_epoch",
                        "monitor": str(early_stopping.monitor if early_stopping is not None else "val_loss"),
                        "mode": str(early_stopping.mode if early_stopping is not None else "min"),
                        "tie_break": "earliest_epoch",
                        "restore_best": True,
                    },
                    "run_mode": run_mode,
                    "requested_device": device,
                    "resolved_device": device_str,
                    "train_subjects": sorted(train_subjects),
                    "val_subjects": sorted(val_subjects),
                    "test_subjects": sorted(test_subjects),
                    "run_dir": str(run_dir),
                },
                save_checkpoint_path,
            )

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
    score_matrix_rows, score_matrix_evidence = _build_score_matrix(eval_window_rows, crop_policy=active_crop_policy, seed=active_seed)
    score_matrix_path = run_dir / "score_matrix.csv"
    _write_score_matrix(score_matrix_path, score_matrix_rows)

    if _is_protocol_single_crop_policy(active_crop_policy):
        eval_trial_rows = _trial_rows_from_score_matrix(score_matrix_rows)

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
    best_epoch = int(checkpoint_payload.get("best_epoch", 0)) if checkpoint_payload is not None else int(result.best_epoch)
    best_metric = float(checkpoint_payload.get("best_metric", float("nan"))) if checkpoint_payload is not None else float(result.best_metric)
    epochs_ran = int(checkpoint_payload.get("epochs_ran", 0)) if checkpoint_payload is not None else len(result.history)
    artifact_training_epochs = int(checkpoint_payload.get("training_epochs", epochs)) if checkpoint_payload is not None else int(epochs)
    artifact_source_epochs = int(checkpoint_payload.get("source_training_epochs", source_epochs)) if checkpoint_payload is not None else int(source_epochs)
    artifact_epochs_overridden = bool(checkpoint_payload.get("training_epochs_overridden", epochs != source_epochs)) if checkpoint_payload is not None else bool(epochs != source_epochs)

    (run_dir / "model_state.local.json").write_text(
        json.dumps({
            "adapter": "torch_classifier",
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "epochs_ran": epochs_ran,
            "training_epochs": artifact_training_epochs,
            "source_training_epochs": artifact_source_epochs,
            "training_epochs_overridden": artifact_epochs_overridden,
            "seed": active_seed,
            "run_mode": run_mode,
            "requested_device": device,
            "resolved_device": device_str,
            "prediction_scope": prediction_scope,
            "evaluation_split": eval_split,
            "score_matrix_evidence": score_matrix_evidence,
            "crop_policy": active_crop_policy,
            "protocol_job_split_manifest": str(Path(split_manifest_path).resolve()) if split_manifest_path is not None else None,
            "checkpoint_reuse": checkpoint_reuse,
            "training_resumed_from_checkpoint": bool(result.resumed_from_checkpoint) if result is not None else False,
            "training_resume_checkpoint": str(run_dir / "training_checkpoint.pt") if checkpoint_payload is None else None,
            "augmentation_transforms": train_transform_configs,
            "augmentation_transform_scope": "train_only" if train_transform_configs else "none",
            "runtime_performance": {
                "torch": torch_runtime_config,
                "dataloader": dataloader_config,
            },
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
    manifest_kwargs: dict[str, Any] = {
        "metrics": metric_report["metrics"],
        "command": command,
        "score_matrix_csv": score_matrix_path,
    }
    if isinstance(active_crop_policy, Mapping):
        manifest_kwargs["crop_policy"] = active_crop_policy
    manifest = build_run_manifest_payload(context, **manifest_kwargs)
    manifest["run_mode"] = run_mode
    manifest["model_name"] = model_name
    manifest["model_kwargs"] = model_kwargs
    manifest["prediction_scope"] = prediction_scope
    manifest["evaluation_split"] = eval_split
    manifest["evaluation_subjects"] = sorted(test_subjects if eval_split == "test" else val_subjects)
    manifest["protocol_job_split_manifest"] = str(Path(split_manifest_path).resolve()) if split_manifest_path is not None else None
    manifest["split_subject_counts"] = {
        "train": len(train_subjects),
        "val": len(val_subjects),
        "test": len(test_subjects),
    }
    manifest["training_epochs"] = artifact_training_epochs
    manifest["source_training_epochs"] = artifact_source_epochs
    manifest["training_epochs_overridden"] = artifact_epochs_overridden
    manifest["requested_device"] = device
    manifest["resolved_device"] = device_str
    manifest["score_matrix_evidence"] = score_matrix_evidence
    manifest["training_resumed_from_checkpoint"] = bool(result.resumed_from_checkpoint) if result is not None else False
    manifest["training_resume_checkpoint"] = str(run_dir / "training_checkpoint.pt") if checkpoint_payload is None else None
    if checkpoint_reuse is not None:
        manifest["checkpoint_reuse"] = checkpoint_reuse
    manifest["augmentation_transforms"] = train_transform_configs
    manifest["augmentation_transform_scope"] = "train_only" if train_transform_configs else "none"
    manifest["declared_protocol"] = str(route_data["evaluation"]["protocol"])
    manifest["adapter_execution_protocol"] = (
        "materialized_protocol_job_split" if split_manifest_path is not None else "single_subject_holdout_split"
    )
    manifest["protocol_evidence_scope"] = (
        "one materialized split/fold from the declared route protocol"
        if split_manifest_path is not None
        else "one adapter-generated subject holdout split; full repeated/nested protocol evidence must be produced by run_evaluation_protocol.py"
    )
    if run_mode in {"full_subjects", "candidate"}:
        ds = yaml.safe_load(dataset_path.read_text(encoding="utf-8")) or {}
        manifest["raw_data_sources"] = ds.get("raw_data_sources") or []
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    monitor.finish("completed")
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
