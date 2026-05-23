"""DEP/HC subject-kind router diagnostics.

This module is intentionally separate from emotion Top-4 route definitions.
It trains a cohort classifier from EEG-derived features and reports whether the
cohort signal generalizes to held-out subjects. Subject identifiers are used
only for split membership, aggregation, and audit output.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


COHORT_TO_LABEL = {"HC": 0, "DEP": 1}
LABEL_TO_COHORT = {value: key for key, value in COHORT_TO_LABEL.items()}
DEFAULT_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)


@dataclass(frozen=True)
class RouterSample:
    x: np.ndarray
    subject_id: str
    trial_id: str
    crop_id: int
    window_start_sec: float
    cohort: str


@dataclass(frozen=True)
class LogisticRouter:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float


def cohort_label(cohort: str) -> int:
    if cohort not in COHORT_TO_LABEL:
        raise ValueError(f"unknown cohort: {cohort}")
    return COHORT_TO_LABEL[cohort]


def covariance_tangent_features(x: np.ndarray, *, eps: float = 1e-3, shrinkage: float = 0.1) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("EEG sample must be shaped [channels, time]")
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError("EEG sample must have at least two channels and two time points")
    centered = arr - arr.mean(axis=1, keepdims=True)
    cov = centered @ centered.T / float(max(centered.shape[1] - 1, 1))
    trace_mean = float(np.trace(cov) / cov.shape[0])
    cov = (1.0 - float(shrinkage)) * cov + float(shrinkage) * trace_mean * np.eye(cov.shape[0])
    cov = cov + float(eps) * np.eye(cov.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, float(eps), None)
    log_cov = (eigvecs * np.log(eigvals)[None, :]) @ eigvecs.T
    rows, cols = np.triu_indices(log_cov.shape[0])
    features = log_cov[rows, cols]
    offdiag = rows != cols
    features[offdiag] *= math.sqrt(2.0)
    return features.astype(np.float64, copy=False)


def bandpower_features(
    x: np.ndarray,
    *,
    sfreq: float = 250.0,
    bands: Sequence[tuple[str, float, float]] = DEFAULT_BANDS,
    eps: float = 1e-12,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("EEG sample must be shaped [channels, time]")
    freqs = np.fft.rfftfreq(arr.shape[1], d=1.0 / float(sfreq))
    spectrum = np.abs(np.fft.rfft(arr - arr.mean(axis=1, keepdims=True), axis=1)) ** 2
    total_mask = freqs > 0
    total_power = spectrum[:, total_mask].sum(axis=1) + float(eps)
    out: list[np.ndarray] = []
    for _name, low, high in bands:
        mask = (freqs >= float(low)) & (freqs < float(high))
        power = spectrum[:, mask].sum(axis=1) + float(eps)
        out.append(np.log(power / total_power))
    return np.concatenate(out).astype(np.float64, copy=False)


def extract_router_features(samples: Sequence[RouterSample], *, feature_set: str = "cov_tangent_bandpower") -> np.ndarray:
    if not samples:
        raise ValueError("at least one sample is required")
    if feature_set not in {"cov_tangent", "bandpower", "cov_tangent_bandpower"}:
        raise ValueError(f"unknown DEP/HC router feature_set: {feature_set}")
    return np.vstack([
        router_features_for_sample(sample.x, feature_set=feature_set)
        for sample in samples
    ])


def router_features_for_sample(x: np.ndarray, *, feature_set: str = "cov_tangent_bandpower") -> np.ndarray:
    if feature_set == "cov_tangent":
        return covariance_tangent_features(x)
    if feature_set == "bandpower":
        return bandpower_features(x)
    if feature_set == "cov_tangent_bandpower":
        return np.concatenate([covariance_tangent_features(x), bandpower_features(x)])
    raise ValueError(f"unknown DEP/HC router feature_set: {feature_set}")


def fit_logistic_router(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.05,
    epochs: int = 600,
    l2: float = 1e-3,
    class_weight_mode: str = "balanced",
) -> LogisticRouter:
    if class_weight_mode not in {"balanced", "uniform"}:
        raise ValueError(f"unknown class_weight_mode: {class_weight_mode}")
    features = np.asarray(x, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels have incompatible shapes")
    if set(np.unique(labels).tolist()) - {0.0, 1.0}:
        raise ValueError("labels must be binary")
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    z = (features - mean) / scale
    weights = np.zeros(z.shape[1], dtype=np.float64)
    bias = 0.0
    n_pos = float(labels.sum())
    n_neg = float(labels.shape[0] - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        raise ValueError("training labels must contain both DEP and HC")
    if class_weight_mode == "balanced":
        class_weight = np.where(labels > 0.5, labels.shape[0] / (2.0 * n_pos), labels.shape[0] / (2.0 * n_neg))
    else:
        class_weight = np.ones_like(labels, dtype=np.float64)
    for _ in range(int(epochs)):
        logits = z @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        err = (probs - labels) * class_weight
        weights -= float(lr) * ((z.T @ err) / labels.shape[0] + float(l2) * weights)
        bias -= float(lr) * float(err.mean())
    return LogisticRouter(mean=mean, scale=scale, weights=weights, bias=float(bias))


def predict_dep_probability(model: LogisticRouter, x: np.ndarray) -> np.ndarray:
    features = np.asarray(x, dtype=np.float64)
    z = (features - model.mean) / model.scale
    logits = z @ model.weights + model.bias
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def balanced_accuracy_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    recalls: list[float] = []
    for label in (0, 1):
        mask = y_t == label
        if not np.any(mask):
            continue
        recalls.append(float(np.mean(y_p[mask] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def brier_score(y_true: np.ndarray, p_dep: np.ndarray) -> float:
    return float(np.mean((np.asarray(p_dep, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)) ** 2))


def evaluate_router(
    train_samples: Sequence[RouterSample],
    eval_samples: Sequence[RouterSample],
    *,
    val_samples: Sequence[RouterSample] | None = None,
    feature_set: str = "cov_tangent_bandpower",
    lr: float = 0.05,
    epochs: int = 600,
    l2: float = 1e-3,
    subject_aggregation: str = "mean",
) -> dict[str, Any]:
    x_train = extract_router_features(train_samples, feature_set=feature_set)
    y_train = np.array([cohort_label(sample.cohort) for sample in train_samples], dtype=int)
    model = fit_logistic_router(x_train, y_train, lr=lr, epochs=epochs, l2=l2)
    threshold = 0.5
    threshold_source = "fixed_0.5"
    if val_samples:
        x_val = extract_router_features(val_samples, feature_set=feature_set)
        y_val = np.array([cohort_label(sample.cohort) for sample in val_samples], dtype=int)
        val_p_dep = predict_dep_probability(model, x_val)
        threshold = select_subject_threshold(val_samples, y_val, val_p_dep, aggregation=subject_aggregation)
        threshold_source = "validation_subjects"

    x_eval = extract_router_features(eval_samples, feature_set=feature_set)
    y_eval = np.array([cohort_label(sample.cohort) for sample in eval_samples], dtype=int)
    p_dep = predict_dep_probability(model, x_eval)
    y_pred = (p_dep >= threshold).astype(int)

    prediction_rows = []
    for sample, truth, prob, pred in zip(eval_samples, y_eval, p_dep, y_pred):
        prediction_rows.append(
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

    subject_rows = aggregate_subject_rows(prediction_rows, threshold=threshold, aggregation=subject_aggregation)
    subject_truth = np.array([cohort_label(row["cohort"]) for row in subject_rows], dtype=int)
    subject_pred = np.array([cohort_label(row["predicted_cohort"]) for row in subject_rows], dtype=int)
    metrics = {
        "window_ba": balanced_accuracy_binary(y_eval, y_pred),
        "subject_ba": balanced_accuracy_binary(subject_truth, subject_pred),
        "window_brier": brier_score(y_eval, p_dep),
        "subject_brier": brier_score(subject_truth, np.array([float(row["subject_score_p_dep"]) for row in subject_rows])),
        "n_train_windows": len(train_samples),
        "n_eval_windows": len(eval_samples),
        "n_eval_subjects": len(subject_rows),
        "feature_dim": int(x_train.shape[1]),
        "feature_set": feature_set,
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "subject_aggregation": subject_aggregation,
    }
    for cohort, label in COHORT_TO_LABEL.items():
        mask = subject_truth == label
        metrics[f"{cohort.lower()}_subject_recall"] = float(np.mean(subject_pred[mask] == label)) if np.any(mask) else float("nan")
    return {"model": model, "metrics": metrics, "prediction_rows": prediction_rows, "subject_rows": subject_rows}


def select_subject_threshold(
    samples: Sequence[RouterSample],
    y_true: np.ndarray,
    p_dep: np.ndarray,
    *,
    objective: str = "balanced_accuracy",
    aggregation: str = "mean",
) -> float:
    summary = subject_threshold_diagnostic(samples, y_true, p_dep, objective=objective, aggregation=aggregation)
    return float(summary["threshold"])


def subject_threshold_diagnostic(
    samples: Sequence[RouterSample],
    y_true: np.ndarray,
    p_dep: np.ndarray,
    *,
    objective: str = "balanced_accuracy",
    aggregation: str = "mean",
) -> dict[str, float | str]:
    if objective not in {"balanced_accuracy", "min_recall", "fixed_0_5", "dep_recall_floor_0p8_hc"}:
        raise ValueError(f"unknown threshold objective: {objective}")
    if aggregation not in {"mean", "median", "trimmed_mean", "vote_frac"}:
        raise ValueError(f"unknown subject aggregation: {aggregation}")
    subject_probs: dict[str, list[float]] = {}
    subject_truth: dict[str, int] = {}
    for sample, truth, prob in zip(samples, y_true, p_dep):
        subject_probs.setdefault(sample.subject_id, []).append(float(prob))
        subject_truth[sample.subject_id] = int(truth)
    if not subject_probs:
        return _threshold_summary(objective=objective, threshold=0.5)
    subjects = sorted(subject_probs)
    subject_scores = _subject_scores_for_threshold(subjects, subject_probs, threshold=0.5, aggregation=aggregation)
    labels = np.array([subject_truth[subject] for subject in subjects], dtype=int)
    if objective == "fixed_0_5":
        metrics = _threshold_recall_metrics(labels, (subject_scores >= 0.5).astype(int))
        return _threshold_summary(objective=objective, threshold=0.5, aggregation=aggregation, **metrics)
    candidates = sorted(set([0.5, *subject_scores.tolist(), *_threshold_boundary_candidates(subject_probs, aggregation=aggregation)]))
    best_threshold = 0.5
    best_key: tuple[float, float, float, float] | None = None
    best_metrics = _threshold_recall_metrics(labels, (subject_scores >= best_threshold).astype(int))
    for threshold in candidates:
        subject_scores = _subject_scores_for_threshold(subjects, subject_probs, threshold=float(threshold), aggregation=aggregation)
        pred = (subject_scores >= float(threshold)).astype(int)
        metrics = _threshold_recall_metrics(labels, pred)
        key = _threshold_selection_key(metrics, objective=objective, threshold=float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return _threshold_summary(objective=objective, threshold=best_threshold, aggregation=aggregation, **best_metrics)


def _threshold_recall_metrics(labels: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    hc_mask = labels == 0
    dep_mask = labels == 1
    hc_recall = float(np.mean(pred[hc_mask] == 0)) if np.any(hc_mask) else float("nan")
    dep_recall = float(np.mean(pred[dep_mask] == 1)) if np.any(dep_mask) else float("nan")
    valid_recalls = [value for value in (hc_recall, dep_recall) if not math.isnan(value)]
    balanced_accuracy = float(np.mean(valid_recalls)) if valid_recalls else float("nan")
    min_recall = float(np.min(valid_recalls)) if valid_recalls else float("nan")
    recall_gap = float(abs(dep_recall - hc_recall)) if len(valid_recalls) == 2 else float("nan")
    return {
        "balanced_accuracy": balanced_accuracy,
        "hc_recall": hc_recall,
        "dep_recall": dep_recall,
        "min_recall": min_recall,
        "recall_gap": recall_gap,
    }


def _threshold_selection_key(metrics: Mapping[str, float], *, objective: str, threshold: float) -> tuple[float, float, float, float]:
    ba = float(metrics["balanced_accuracy"])
    min_recall = float(metrics["min_recall"])
    recall_gap = float(metrics["recall_gap"])
    hc_recall = float(metrics["hc_recall"])
    dep_recall = float(metrics["dep_recall"])
    if objective == "balanced_accuracy":
        primary = ba
        secondary = min_recall
        tertiary = -recall_gap
    elif objective == "min_recall":
        primary = min_recall
        secondary = ba
        tertiary = -recall_gap
    elif objective == "dep_recall_floor_0p8_hc":
        passes_floor = float(dep_recall >= 0.8 * hc_recall)
        primary = passes_floor
        secondary = ba
        tertiary = hc_recall
    elif objective == "fixed_0_5":
        primary = ba
        secondary = min_recall
        tertiary = -recall_gap
    else:
        raise ValueError(f"unknown threshold objective: {objective}")
    return (primary, secondary, tertiary, -abs(float(threshold) - 0.5))


def _threshold_summary(
    *,
    objective: str,
    threshold: float,
    aggregation: str = "mean",
    balanced_accuracy: float = float("nan"),
    hc_recall: float = float("nan"),
    dep_recall: float = float("nan"),
    min_recall: float = float("nan"),
    recall_gap: float = float("nan"),
) -> dict[str, float | str]:
    return {
        "objective": objective,
        "threshold": float(threshold),
        "aggregation": aggregation,
        "balanced_accuracy": float(balanced_accuracy),
        "hc_recall": float(hc_recall),
        "dep_recall": float(dep_recall),
        "min_recall": float(min_recall),
        "recall_gap": float(recall_gap),
    }


def _subject_probability_score(probabilities: Sequence[float], *, threshold: float, aggregation: str) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    if aggregation == "mean":
        return float(np.mean(values))
    if aggregation == "median":
        return float(np.median(values))
    if aggregation == "trimmed_mean":
        if values.size >= 3:
            sorted_values = np.sort(values)
            return float(np.mean(sorted_values[1:-1]))
        return float(np.mean(values))
    if aggregation == "vote_frac":
        return float(np.mean(values >= float(threshold)))
    raise ValueError(f"unknown subject aggregation: {aggregation}")


def _subject_scores_for_threshold(
    subjects: Sequence[str],
    subject_probs: Mapping[str, Sequence[float]],
    *,
    threshold: float,
    aggregation: str,
) -> np.ndarray:
    return np.array(
        [
            _subject_probability_score(subject_probs[subject], threshold=threshold, aggregation=aggregation)
            for subject in subjects
        ],
        dtype=np.float64,
    )


def _threshold_boundary_candidates(subject_probs: Mapping[str, Sequence[float]], *, aggregation: str) -> list[float]:
    if aggregation != "vote_frac":
        return []
    raw_probs = [float(prob) for probs in subject_probs.values() for prob in probs]
    vote_boundaries = {
        float(k) / float(len(probs))
        for probs in subject_probs.values()
        for k in range(0, len(probs) + 1)
        if len(probs) > 0
    }
    return [*raw_probs, *vote_boundaries]


def aggregate_subject_rows(
    prediction_rows: Sequence[Mapping[str, str]],
    *,
    threshold: float = 0.5,
    aggregation: str = "mean",
) -> list[dict[str, str]]:
    if aggregation not in {"mean", "median", "trimmed_mean", "vote_frac"}:
        raise ValueError(f"unknown subject aggregation: {aggregation}")
    by_subject: dict[str, list[Mapping[str, str]]] = {}
    for row in prediction_rows:
        by_subject.setdefault(str(row["subject_id"]), []).append(row)
    out: list[dict[str, str]] = []
    for subject_id, rows in sorted(by_subject.items()):
        probabilities = np.array([float(row["p_dep"]) for row in rows], dtype=np.float64)
        score_p_dep = _subject_probability_score(probabilities, threshold=threshold, aggregation=aggregation)
        pred_label = int(score_p_dep >= float(threshold))
        truth = str(rows[0]["y_true"])
        cohort = str(rows[0]["cohort"])
        out.append(
            {
                "subject_id": subject_id,
                "cohort": cohort,
                "y_true": truth,
                "mean_p_dep": f"{float(np.mean(probabilities)):.12g}",
                "mean_p_hc": f"{1.0 - float(np.mean(probabilities)):.12g}",
                "subject_score_p_dep": f"{score_p_dep:.12g}",
                "subject_aggregation": aggregation,
                "threshold": f"{float(threshold):.12g}",
                "confidence": f"{max(score_p_dep, 1.0 - score_p_dep):.12g}",
                "predicted_cohort": LABEL_TO_COHORT[pred_label],
                "correct": str(int(pred_label == int(truth))),
                "n_windows": str(len(rows)),
            }
        )
    return out


def write_router_outputs(result: Mapping[str, Any], out_dir: Path, *, config: Mapping[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = list(result["prediction_rows"])
    subject_rows = list(result["subject_rows"])
    if prediction_rows:
        with (out_dir / "router_predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0]))
            writer.writeheader()
            writer.writerows(prediction_rows)
    if subject_rows:
        with (out_dir / "subject_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(subject_rows[0]))
            writer.writeheader()
            writer.writerows(subject_rows)
    payload = {"task": "dep_hc_router", "config": dict(config), "metrics": result["metrics"]}
    (out_dir / "router_diagnostic.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "router_diagnostic.md").write_text(render_router_markdown(payload), encoding="utf-8")


def render_router_markdown(payload: Mapping[str, Any]) -> str:
    metrics = payload.get("metrics", {})
    lines = [
        "# DEP/HC Router Diagnostic",
        "",
        "This is a cohort-classification diagnostic, separate from emotion Top-4 route evidence.",
        "Subject identifiers are used for split membership and reporting only.",
        "",
        "## Metrics",
        "",
    ]
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            lines.append(f"- `{key}`: {value:.6f}")
        else:
            lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)
