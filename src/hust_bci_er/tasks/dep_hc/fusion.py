"""Score-level fusion helpers for independent DEP/HC feature branches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np

from hust_bci_er.analysis.dep_hc_router import (
    RouterSample,
    aggregate_subject_rows,
    balanced_accuracy_binary,
    brier_score,
    cohort_label,
    crop_combo_subject_metrics,
    fit_logistic_router,
    predict_dep_probability,
    subject_threshold_diagnostic,
)
from hust_bci_er.tasks.dep_hc.experiment import dep_hc_prediction_rows, extract_dep_hc_task_features
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
    fixed_first_weight: float | None = None,
    subject_aggregation: str = "mean",
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
        fixed_first_weight=fixed_first_weight,
        subject_aggregation=subject_aggregation,
    )
    weights = np.asarray([weight, 1.0 - weight], dtype=np.float64)
    threshold = float(threshold_summary["threshold"])
    p_dep = _weighted_probabilities(eval_probs, weights)
    y_pred = (p_dep >= threshold).astype(int)

    prediction_rows = dep_hc_prediction_rows(eval_samples, y_eval, p_dep, y_pred)
    subject_rows = aggregate_subject_rows(prediction_rows, threshold=threshold, aggregation=subject_aggregation)
    subject_truth = np.array([cohort_label(row["cohort"]) for row in subject_rows], dtype=int)
    subject_pred = np.array([cohort_label(row["predicted_cohort"]) for row in subject_rows], dtype=int)
    metrics: dict[str, Any] = {
        "task": "dep_hc",
        "fusion_feature_sets": list(features),
        "fusion_weight_by_feature": {features[0]: float(weights[0]), features[1]: float(weights[1])},
        "fusion_weight_source": "fixed" if fixed_first_weight is not None else "validation_subjects",
        "feature_dims": feature_dims,
        "window_ba": balanced_accuracy_binary(y_eval, y_pred),
        "subject_ba": balanced_accuracy_binary(subject_truth, subject_pred),
        "window_brier": brier_score(y_eval, p_dep),
        "subject_brier": brier_score(subject_truth, np.array([float(row["subject_score_p_dep"]) for row in subject_rows])),
        "n_train_windows": len(train_samples),
        "n_eval_windows": len(eval_samples),
        "n_eval_subjects": len(subject_rows),
        "threshold": threshold,
        "threshold_source": "fixed_0.5" if threshold_objective == "fixed_0_5" else "validation_subjects",
        "threshold_objective": threshold_objective,
        "class_weight_mode": class_weight_mode,
        "subject_aggregation": subject_aggregation,
    }
    for key, value in threshold_summary.items():
        if key in {"objective", "threshold"}:
            continue
        metrics[f"validation_threshold_{key}"] = value if isinstance(value, str) else float(value)
    for cohort, label in {"HC": 0, "DEP": 1}.items():
        mask = subject_truth == label
        metrics[f"{cohort.lower()}_subject_recall"] = float(np.mean(subject_pred[mask] == label)) if np.any(mask) else float("nan")
    metrics.update(crop_combo_subject_metrics(prediction_rows, threshold=threshold, aggregation=subject_aggregation))
    return {"metrics": metrics, "prediction_rows": prediction_rows, "subject_rows": subject_rows}


def _select_fusion_weight_and_threshold(
    samples: Sequence[RouterSample],
    y_true: np.ndarray,
    probabilities: Sequence[np.ndarray],
    *,
    threshold_objective: str,
    weight_step: float,
    fixed_first_weight: float | None = None,
    subject_aggregation: str,
) -> tuple[float, dict[str, float | str]]:
    if fixed_first_weight is not None:
        fixed_weight = float(fixed_first_weight)
        if fixed_weight < 0.0 or fixed_weight > 1.0:
            raise ValueError("fixed_first_weight must be in [0, 1]")
        weights = np.asarray([fixed_weight, 1.0 - fixed_weight], dtype=np.float64)
        p_dep = _weighted_probabilities(probabilities, weights)
        summary = subject_threshold_diagnostic(
            samples,
            y_true,
            p_dep,
            objective=threshold_objective,
            aggregation=subject_aggregation,
        )
        return fixed_weight, summary

    best_weight = 0.5
    best_summary: dict[str, float | str] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for weight in _weight_candidates(weight_step):
        weights = np.asarray([weight, 1.0 - weight], dtype=np.float64)
        p_dep = _weighted_probabilities(probabilities, weights)
        summary = subject_threshold_diagnostic(
            samples,
            y_true,
            p_dep,
            objective=threshold_objective,
            aggregation=subject_aggregation,
        )
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
    if threshold_objective in {"balanced_accuracy", "fixed_0_5"}:
        primary = ba
        secondary = min_recall
        tertiary = -recall_gap
    elif threshold_objective == "min_recall":
        primary = min_recall
        secondary = ba
        tertiary = -recall_gap
    elif threshold_objective == "dep_recall_floor_0p8_hc":
        hc_recall = float(summary["hc_recall"])
        dep_recall = float(summary["dep_recall"])
        primary = float(dep_recall >= 0.8 * hc_recall)
        secondary = ba
        tertiary = hc_recall
    else:
        raise ValueError(f"unknown threshold objective: {threshold_objective}")
    return (primary, secondary, tertiary, -abs(float(weight) - 0.5))


def _weighted_probabilities(probabilities: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    stacked = np.vstack([np.asarray(prob, dtype=np.float64) for prob in probabilities])
    return np.clip(weights @ stacked, 0.0, 1.0)


PREDICTION_KEY_FIELDS = ("subject_id", "trial_id", "crop_id", "window_start_sec")


def evaluate_dep_hc_score_fusion(
    component_prediction_rows: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    val_component_prediction_rows: Mapping[str, Sequence[Mapping[str, str]]] | None = None,
    weights: Mapping[str, float] | None = None,
    threshold_objective: str = "fixed_0_5",
    subject_aggregation: str = "vote_frac",
    select_two_way_weight: bool = False,
    weight_step: float = 0.05,
) -> dict[str, Any]:
    """Fuse already materialized DEP/HC prediction probabilities.

    Existing prediction rows are aligned by subject/trial/crop/window metadata.
    The metadata is used only for row alignment, aggregation, and audit output;
    component scores are the only fused values.
    """

    names = tuple(str(name) for name in component_prediction_rows)
    if len(names) < 2:
        raise ValueError("score fusion requires at least two component prediction tables")
    aligned_eval = _align_prediction_components(component_prediction_rows)
    aligned_val = _align_prediction_components(val_component_prediction_rows) if val_component_prediction_rows else None
    y_eval = _labels_from_aligned(aligned_eval, names[0])

    if weights is not None and select_two_way_weight:
        raise ValueError("fixed weights and select_two_way_weight are mutually exclusive")
    if weights is not None:
        weight_vector = _normalize_named_weights(names, weights)
        weight_source = "fixed"
    elif select_two_way_weight:
        if aligned_val is None:
            raise ValueError("select_two_way_weight requires validation component predictions")
        if len(names) != 2:
            raise ValueError("select_two_way_weight currently requires exactly two components")
        if weight_step <= 0.0 or weight_step > 1.0:
            raise ValueError("weight_step must be in (0, 1]")
        weight, threshold_summary = _select_prediction_fusion_weight_and_threshold(
            aligned_val,
            names,
            threshold_objective=threshold_objective,
            subject_aggregation=subject_aggregation,
            weight_step=weight_step,
        )
        weight_vector = np.asarray([weight, 1.0 - weight], dtype=np.float64)
        weight_source = "validation_predictions"
    else:
        weight_vector = np.full(len(names), 1.0 / float(len(names)), dtype=np.float64)
        weight_source = "equal_fixed"

    if not select_two_way_weight:
        threshold_summary = _prediction_threshold_summary(
            aligned_val,
            names,
            weight_vector,
            threshold_objective=threshold_objective,
            subject_aggregation=subject_aggregation,
        )
    threshold = float(threshold_summary["threshold"])
    p_dep = _fused_probabilities(aligned_eval, names, weight_vector)
    y_pred = (p_dep >= threshold).astype(int)
    prediction_rows = _fused_prediction_rows(aligned_eval, names[0], p_dep, y_pred)
    subject_rows = aggregate_subject_rows(prediction_rows, threshold=threshold, aggregation=subject_aggregation)
    subject_truth = np.array([cohort_label(row["cohort"]) for row in subject_rows], dtype=int)
    subject_pred = np.array([cohort_label(row["predicted_cohort"]) for row in subject_rows], dtype=int)
    metrics: dict[str, Any] = {
        "task": "dep_hc",
        "score_fusion_components": list(names),
        "score_fusion_weights_by_component": {name: float(weight_vector[index]) for index, name in enumerate(names)},
        "fusion_weight_source": weight_source,
        "window_ba": balanced_accuracy_binary(y_eval, y_pred),
        "subject_ba": balanced_accuracy_binary(subject_truth, subject_pred),
        "window_brier": brier_score(y_eval, p_dep),
        "subject_brier": brier_score(subject_truth, np.array([float(row["subject_score_p_dep"]) for row in subject_rows])),
        "n_eval_windows": len(p_dep),
        "n_eval_subjects": len(subject_rows),
        "threshold": threshold,
        "threshold_source": "fixed_0.5" if threshold_objective == "fixed_0_5" else "validation_predictions",
        "threshold_objective": threshold_objective,
        "subject_aggregation": subject_aggregation,
    }
    for key, value in threshold_summary.items():
        if key in {"objective", "threshold"}:
            continue
        metrics[f"validation_threshold_{key}"] = value if isinstance(value, str) else float(value)
    for cohort, label in {"HC": 0, "DEP": 1}.items():
        mask = subject_truth == label
        metrics[f"{cohort.lower()}_subject_recall"] = float(np.mean(subject_pred[mask] == label)) if np.any(mask) else float("nan")
    metrics.update(crop_combo_subject_metrics(prediction_rows, threshold=threshold, aggregation=subject_aggregation))
    return {"metrics": metrics, "prediction_rows": prediction_rows, "subject_rows": subject_rows}


def _prediction_threshold_summary(
    aligned_val: dict[str, list[Mapping[str, str]]] | None,
    names: Sequence[str],
    weights: np.ndarray,
    *,
    threshold_objective: str,
    subject_aggregation: str,
) -> dict[str, float | str]:
    if aligned_val is None:
        if threshold_objective != "fixed_0_5":
            raise ValueError("validation component predictions are required for non-fixed score-fusion thresholds")
        return {
            "objective": "fixed_0_5",
            "threshold": 0.5,
            "balanced_accuracy": float("nan"),
            "hc_recall": float("nan"),
            "dep_recall": float("nan"),
            "min_recall": float("nan"),
            "recall_gap": float("nan"),
        }
    p_dep = _fused_probabilities(aligned_val, names, weights)
    samples = _samples_from_aligned(aligned_val, names[0])
    labels = _labels_from_aligned(aligned_val, names[0])
    return subject_threshold_diagnostic(
        samples,
        labels,
        p_dep,
        objective=threshold_objective,
        aggregation=subject_aggregation,
    )


def _select_prediction_fusion_weight_and_threshold(
    aligned_val: dict[str, list[Mapping[str, str]]],
    names: Sequence[str],
    *,
    threshold_objective: str,
    subject_aggregation: str,
    weight_step: float,
) -> tuple[float, dict[str, float | str]]:
    samples = _samples_from_aligned(aligned_val, names[0])
    labels = _labels_from_aligned(aligned_val, names[0])
    best_weight = 0.5
    best_summary: dict[str, float | str] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for weight in _weight_candidates(weight_step):
        weights = np.asarray([weight, 1.0 - weight], dtype=np.float64)
        p_dep = _fused_probabilities(aligned_val, names, weights)
        summary = subject_threshold_diagnostic(
            samples,
            labels,
            p_dep,
            objective=threshold_objective,
            aggregation=subject_aggregation,
        )
        key = _fusion_selection_key(summary, threshold_objective=threshold_objective, weight=float(weight))
        if best_key is None or key > best_key:
            best_key = key
            best_weight = float(weight)
            best_summary = summary
    if best_summary is None:
        raise RuntimeError("failed to select score-fusion weight")
    return best_weight, best_summary


def _align_prediction_components(
    component_rows: Mapping[str, Sequence[Mapping[str, str]]] | None,
) -> dict[str, list[Mapping[str, str]]]:
    if not component_rows:
        raise ValueError("component prediction rows are required")
    aligned: dict[str, list[Mapping[str, str]]] = {}
    reference_keys: list[tuple[str, str, str, str]] | None = None
    reference_meta: dict[tuple[str, str, str, str], tuple[str, str]] = {}
    for name, rows in component_rows.items():
        row_by_key: dict[tuple[str, str, str, str], Mapping[str, str]] = {}
        for row in rows:
            key = tuple(str(row.get(field, "")) for field in PREDICTION_KEY_FIELDS)
            if any(not part for part in key):
                raise ValueError(f"component {name} has a prediction row with incomplete alignment metadata")
            if key in row_by_key:
                raise ValueError(f"component {name} has duplicate prediction key: {key}")
            if "p_dep" not in row:
                raise ValueError(f"component {name} prediction row is missing p_dep")
            row_by_key[key] = row
        keys = sorted(row_by_key)
        if reference_keys is None:
            reference_keys = keys
            reference_meta = {key: (str(row_by_key[key].get("cohort", "")), str(row_by_key[key].get("y_true", ""))) for key in keys}
        elif keys != reference_keys:
            raise ValueError(f"component {name} prediction keys do not match the reference component")
        for key in keys:
            meta = (str(row_by_key[key].get("cohort", "")), str(row_by_key[key].get("y_true", "")))
            if meta != reference_meta[key]:
                raise ValueError(f"component {name} label metadata does not match reference for key {key}")
        aligned[str(name)] = [row_by_key[key] for key in keys]
    return aligned


def _normalize_named_weights(names: Sequence[str], weights: Mapping[str, float]) -> np.ndarray:
    missing = [name for name in names if name not in weights]
    extra = sorted(set(weights) - set(names))
    if missing or extra:
        raise ValueError(f"score-fusion weights must match components, missing={missing}, extra={extra}")
    arr = np.asarray([float(weights[name]) for name in names], dtype=np.float64)
    if np.any(arr < 0.0):
        raise ValueError("score-fusion weights must be non-negative")
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("score-fusion weights must sum to a positive value")
    return arr / total


def _fused_probabilities(
    aligned: dict[str, list[Mapping[str, str]]],
    names: Sequence[str],
    weights: np.ndarray,
) -> np.ndarray:
    stacked = np.vstack([[float(row["p_dep"]) for row in aligned[name]] for name in names])
    return np.clip(weights @ stacked, 0.0, 1.0)


def _labels_from_aligned(aligned: dict[str, list[Mapping[str, str]]], name: str) -> np.ndarray:
    return np.asarray([int(row["y_true"]) for row in aligned[name]], dtype=int)


def _samples_from_aligned(aligned: dict[str, list[Mapping[str, str]]], name: str) -> list[RouterSample]:
    samples: list[RouterSample] = []
    for row in aligned[name]:
        samples.append(
            RouterSample(
                x=np.empty((0, 0), dtype=np.float32),
                subject_id=str(row["subject_id"]),
                trial_id=str(row["trial_id"]),
                crop_id=int(float(row["crop_id"])),
                window_start_sec=float(row["window_start_sec"]),
                cohort=str(row["cohort"]),
            )
        )
    return samples


def _fused_prediction_rows(
    aligned: dict[str, list[Mapping[str, str]]],
    reference_name: str,
    p_dep: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for reference, prob, pred in zip(aligned[reference_name], p_dep, y_pred):
        rows.append(
            {
                "subject_id": str(reference["subject_id"]),
                "trial_id": str(reference["trial_id"]),
                "crop_id": str(reference["crop_id"]),
                "window_start_sec": str(reference["window_start_sec"]),
                "cohort": str(reference["cohort"]),
                "y_true": str(int(reference["y_true"])),
                "p_hc": f"{1.0 - float(prob):.12g}",
                "p_dep": f"{float(prob):.12g}",
                "confidence": f"{max(float(prob), 1.0 - float(prob)):.12g}",
                "y_pred": str(int(pred)),
                "predicted_cohort": "DEP" if int(pred) == 1 else "HC",
            }
        )
    return rows


