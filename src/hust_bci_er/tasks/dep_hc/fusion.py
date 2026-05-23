"""Score-level fusion helpers for independent DEP/HC feature branches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from hust_bci_er.analysis.dep_hc_router import (
    LABEL_TO_COHORT,
    RouterSample,
    aggregate_subject_rows,
    balanced_accuracy_binary,
    brier_score,
    cohort_label,
    fit_logistic_router,
    predict_dep_probability,
    subject_threshold_diagnostic,
)
from hust_bci_er.tasks.dep_hc.experiment import extract_dep_hc_task_features
from hust_bci_er.tasks.dep_hc.features import DEP_HC_FEATURE_SETS


def evaluate_dep_hc_feature_fusion(
    train_samples: Sequence[RouterSample],
    eval_samples: Sequence[RouterSample],
    *,
    val_samples: Sequence[RouterSample],
    feature_sets: Sequence[str],
    sfreq: float = 250.0,
    channel_montage: str | None = "hust_30_a2",
    lr: float = 0.05,
    epochs: int = 600,
    l2: float = 1e-3,
    class_weight_mode: str = "balanced",
    threshold_objective: str = "balanced_accuracy",
    weight_step: float = 0.05,
) -> dict[str, Any]:
    features = tuple(str(feature_set) for feature_set in feature_sets)
    if len(features) != 2:
        raise ValueError("DEP/HC feature fusion currently requires exactly two feature sets")
    unknown = set(features) - set(DEP_HC_FEATURE_SETS)
    if unknown:
        raise ValueError(f"unknown DEP/HC task feature_set: {sorted(unknown)}")
    if not val_samples:
        raise ValueError("feature fusion requires validation subjects for weight and threshold selection")
    if weight_step <= 0.0 or weight_step > 1.0:
        raise ValueError("weight_step must be in (0, 1]")

    y_train = np.array([cohort_label(sample.cohort) for sample in train_samples], dtype=int)
    y_val = np.array([cohort_label(sample.cohort) for sample in val_samples], dtype=int)
    y_eval = np.array([cohort_label(sample.cohort) for sample in eval_samples], dtype=int)
    val_probs: list[np.ndarray] = []
    eval_probs: list[np.ndarray] = []
    feature_dims: dict[str, int] = {}
    for feature_set in features:
        x_train = extract_dep_hc_task_features(
            train_samples,
            feature_set=feature_set,
            sfreq=sfreq,
            channel_montage=channel_montage,
        )
        model = fit_logistic_router(
            x_train,
            y_train,
            lr=lr,
            epochs=epochs,
            l2=l2,
            class_weight_mode=class_weight_mode,
        )
        x_val = extract_dep_hc_task_features(
            val_samples,
            feature_set=feature_set,
            sfreq=sfreq,
            channel_montage=channel_montage,
        )
        x_eval = extract_dep_hc_task_features(
            eval_samples,
            feature_set=feature_set,
            sfreq=sfreq,
            channel_montage=channel_montage,
        )
        val_probs.append(predict_dep_probability(model, x_val))
        eval_probs.append(predict_dep_probability(model, x_eval))
        feature_dims[feature_set] = int(x_train.shape[1])

    weight, threshold_summary = _select_fusion_weight_and_threshold(
        val_samples,
        y_val,
        val_probs,
        threshold_objective=threshold_objective,
        weight_step=weight_step,
    )
    weights = np.asarray([weight, 1.0 - weight], dtype=np.float64)
    threshold = float(threshold_summary["threshold"])
    p_dep = _weighted_probabilities(eval_probs, weights)
    y_pred = (p_dep >= threshold).astype(int)

    prediction_rows = _prediction_rows(eval_samples, y_eval, p_dep, y_pred)
    subject_rows = aggregate_subject_rows(prediction_rows, threshold=threshold)
    subject_truth = np.array([cohort_label(row["cohort"]) for row in subject_rows], dtype=int)
    subject_pred = np.array([cohort_label(row["predicted_cohort"]) for row in subject_rows], dtype=int)
    metrics: dict[str, Any] = {
        "task": "dep_hc",
        "fusion_feature_sets": list(features),
        "fusion_weight_by_feature": {features[0]: float(weights[0]), features[1]: float(weights[1])},
        "feature_dims": feature_dims,
        "window_ba": balanced_accuracy_binary(y_eval, y_pred),
        "subject_ba": balanced_accuracy_binary(subject_truth, subject_pred),
        "window_brier": brier_score(y_eval, p_dep),
        "subject_brier": brier_score(subject_truth, np.array([float(row["mean_p_dep"]) for row in subject_rows])),
        "n_train_windows": len(train_samples),
        "n_eval_windows": len(eval_samples),
        "n_eval_subjects": len(subject_rows),
        "threshold": threshold,
        "threshold_source": "validation_subjects",
        "threshold_objective": threshold_objective,
        "class_weight_mode": class_weight_mode,
    }
    for key, value in threshold_summary.items():
        if key in {"objective", "threshold"}:
            continue
        metrics[f"validation_threshold_{key}"] = float(value)
    for cohort, label in {"HC": 0, "DEP": 1}.items():
        mask = subject_truth == label
        metrics[f"{cohort.lower()}_subject_recall"] = float(np.mean(subject_pred[mask] == label)) if np.any(mask) else float("nan")
    return {"metrics": metrics, "prediction_rows": prediction_rows, "subject_rows": subject_rows}


def _select_fusion_weight_and_threshold(
    samples: Sequence[RouterSample],
    y_true: np.ndarray,
    probabilities: Sequence[np.ndarray],
    *,
    threshold_objective: str,
    weight_step: float,
) -> tuple[float, dict[str, float | str]]:
    best_weight = 0.5
    best_summary: dict[str, float | str] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for weight in _weight_candidates(weight_step):
        weights = np.asarray([weight, 1.0 - weight], dtype=np.float64)
        p_dep = _weighted_probabilities(probabilities, weights)
        summary = subject_threshold_diagnostic(samples, y_true, p_dep, objective=threshold_objective)
        key = _fusion_selection_key(summary, threshold_objective=threshold_objective, weight=float(weight))
        if best_key is None or key > best_key:
            best_key = key
            best_weight = float(weight)
            best_summary = summary
    if best_summary is None:
        raise RuntimeError("failed to select DEP/HC fusion weight")
    return best_weight, best_summary


def _weight_candidates(weight_step: float) -> tuple[float, ...]:
    values = {0.0, 1.0}
    weight = 0.0
    while weight < 1.0:
        values.add(round(float(weight), 12))
        weight += float(weight_step)
    return tuple(sorted(values))


def _fusion_selection_key(
    summary: dict[str, float | str],
    *,
    threshold_objective: str,
    weight: float,
) -> tuple[float, float, float, float]:
    ba = float(summary["balanced_accuracy"])
    min_recall = float(summary["min_recall"])
    recall_gap = float(summary["recall_gap"])
    if threshold_objective == "balanced_accuracy":
        primary = ba
        secondary = min_recall
        tertiary = -recall_gap
    elif threshold_objective == "min_recall":
        primary = min_recall
        secondary = ba
        tertiary = -recall_gap
    else:
        raise ValueError(f"unknown threshold objective: {threshold_objective}")
    return (primary, secondary, tertiary, -abs(float(weight) - 0.5))


def _weighted_probabilities(probabilities: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    stacked = np.vstack([np.asarray(prob, dtype=np.float64) for prob in probabilities])
    return np.clip(weights @ stacked, 0.0, 1.0)


def _prediction_rows(
    samples: Sequence[RouterSample],
    y_true: np.ndarray,
    p_dep: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sample, truth, prob, pred in zip(samples, y_true, p_dep, y_pred):
        rows.append(
            {
                "subject_id": sample.subject_id,
                "trial_id": sample.trial_id,
                "crop_id": str(sample.crop_id),
                "window_start_sec": f"{float(sample.window_start_sec):.8f}",
                "cohort": sample.cohort,
                "y_true": str(int(truth)),
                "p_hc": f"{1.0 - float(prob):.12g}",
                "p_dep": f"{float(prob):.12g}",
                "confidence": f"{max(float(prob), 1.0 - float(prob)):.12g}",
                "y_pred": str(int(pred)),
                "predicted_cohort": LABEL_TO_COHORT[int(pred)],
            }
        )
    return rows
