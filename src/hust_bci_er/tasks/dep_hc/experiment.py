"""Experiment helpers for independent DEP/HC logistic feature branches."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from hust_bci_er.analysis.dep_hc_router import (
    LABEL_TO_COHORT,
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
from hust_bci_er.tasks.dep_hc.features import DEP_HC_FEATURE_SETS, dep_hc_features

DEP_HC_CLASSIFIERS = {"logistic", "linear_svm", "random_forest"}


def extract_dep_hc_task_features(
    samples: Sequence[RouterSample],
    *,
    feature_set: str = "traditional",
    sfreq: float = 250.0,
    channel_montage: str | None = "hust_30_a2",
) -> np.ndarray:
    if not samples:
        raise ValueError("at least one sample is required")
    if feature_set not in DEP_HC_FEATURE_SETS:
        raise ValueError(f"unknown DEP/HC task feature_set: {feature_set}")
    return np.vstack(
        [
            dep_hc_features(sample.x, feature_set=feature_set, sfreq=sfreq, channel_montage=channel_montage)
            for sample in samples
        ]
    )


def dep_hc_prediction_rows(
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


def evaluate_dep_hc_task(
    train_samples: Sequence[RouterSample],
    eval_samples: Sequence[RouterSample],
    *,
    val_samples: Sequence[RouterSample] | None = None,
    feature_set: str = "traditional",
    sfreq: float = 250.0,
    channel_montage: str | None = "hust_30_a2",
    lr: float = 0.05,
    epochs: int = 600,
    l2: float = 1e-3,
    threshold_objective: str = "balanced_accuracy",
    class_weight_mode: str = "balanced",
    subject_aggregation: str = "mean",
    classifier: str = "logistic",
    random_state: int = 42,
) -> dict[str, Any]:
    x_train = extract_dep_hc_task_features(
        train_samples,
        feature_set=feature_set,
        sfreq=sfreq,
        channel_montage=channel_montage,
    )
    y_train = np.array([cohort_label(sample.cohort) for sample in train_samples], dtype=int)
    model = fit_dep_hc_classifier(
        x_train,
        y_train,
        classifier=classifier,
        lr=lr,
        epochs=epochs,
        l2=l2,
        class_weight_mode=class_weight_mode,
        random_state=random_state,
    )
    threshold = 0.5
    threshold_source = "fixed_0.5"
    if val_samples:
        x_val = extract_dep_hc_task_features(
            val_samples,
            feature_set=feature_set,
            sfreq=sfreq,
            channel_montage=channel_montage,
        )
        y_val = np.array([cohort_label(sample.cohort) for sample in val_samples], dtype=int)
        val_p_dep = predict_dep_hc_probability(model, x_val, classifier=classifier)
        threshold_summary = subject_threshold_diagnostic(
            val_samples,
            y_val,
            val_p_dep,
            objective=threshold_objective,
            aggregation=subject_aggregation,
        )
        threshold = float(threshold_summary["threshold"])
        threshold_source = "fixed_0.5" if threshold_objective == "fixed_0_5" else "validation_subjects"
    else:
        threshold_summary = {
            "objective": threshold_objective,
            "threshold": float(threshold),
            "balanced_accuracy": float("nan"),
            "hc_recall": float("nan"),
            "dep_recall": float("nan"),
            "min_recall": float("nan"),
            "recall_gap": float("nan"),
        }

    x_eval = extract_dep_hc_task_features(
        eval_samples,
        feature_set=feature_set,
        sfreq=sfreq,
        channel_montage=channel_montage,
    )
    y_eval = np.array([cohort_label(sample.cohort) for sample in eval_samples], dtype=int)
    p_dep = predict_dep_hc_probability(model, x_eval, classifier=classifier)
    y_pred = (p_dep >= threshold).astype(int)

    prediction_rows = dep_hc_prediction_rows(eval_samples, y_eval, p_dep, y_pred)

    subject_rows = aggregate_subject_rows(prediction_rows, threshold=threshold, aggregation=subject_aggregation)
    subject_truth = np.array([cohort_label(row["cohort"]) for row in subject_rows], dtype=int)
    subject_pred = np.array([cohort_label(row["predicted_cohort"]) for row in subject_rows], dtype=int)
    metrics = {
        "task": "dep_hc",
        "window_ba": balanced_accuracy_binary(y_eval, y_pred),
        "subject_ba": balanced_accuracy_binary(subject_truth, subject_pred),
        "window_brier": brier_score(y_eval, p_dep),
        "subject_brier": brier_score(subject_truth, np.array([float(row["subject_score_p_dep"]) for row in subject_rows])),
        "n_train_windows": len(train_samples),
        "n_eval_windows": len(eval_samples),
        "n_eval_subjects": len(subject_rows),
        "feature_dim": int(x_train.shape[1]),
        "feature_set": feature_set,
        "classifier": classifier,
        "threshold": float(threshold),
        "threshold_source": threshold_source,
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
    return {"model": model, "metrics": metrics, "prediction_rows": prediction_rows, "subject_rows": subject_rows}


def fit_dep_hc_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    classifier: str = "logistic",
    lr: float = 0.05,
    epochs: int = 600,
    l2: float = 1e-3,
    class_weight_mode: str = "balanced",
    random_state: int = 42,
) -> Any:
    if classifier not in DEP_HC_CLASSIFIERS:
        raise ValueError(f"unknown DEP/HC classifier: {classifier}")
    if classifier == "logistic":
        return fit_logistic_router(
            x_train,
            y_train,
            lr=lr,
            epochs=epochs,
            l2=l2,
            class_weight_mode=class_weight_mode,
        )
    if class_weight_mode not in {"balanced", "uniform"}:
        raise ValueError(f"unknown class_weight_mode: {class_weight_mode}")
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional local environment.
        raise ModuleNotFoundError("scikit-learn is required for DEP/HC sklearn classifiers") from exc
    class_weight = "balanced" if class_weight_mode == "balanced" else None
    if classifier == "linear_svm":
        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel="linear",
                C=1.0,
                probability=True,
                class_weight=class_weight,
                random_state=int(random_state),
            ),
        ).fit(x_train, y_train)
    if classifier == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight=class_weight,
            random_state=int(random_state),
            n_jobs=1,
        ).fit(x_train, y_train)
    raise ValueError(f"unknown DEP/HC classifier: {classifier}")


def predict_dep_hc_probability(model: Any, x: np.ndarray, *, classifier: str = "logistic") -> np.ndarray:
    if classifier == "logistic":
        return predict_dep_probability(model, x)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    raise TypeError(f"DEP/HC classifier does not expose predict_proba: {classifier}")


def write_dep_hc_task_outputs(result: dict[str, Any], out_dir: Path, *, config: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = list(result["prediction_rows"])
    subject_rows = list(result["subject_rows"])
    if prediction_rows:
        with (out_dir / "dep_hc_predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0]))
            writer.writeheader()
            writer.writerows(prediction_rows)
    if subject_rows:
        with (out_dir / "dep_hc_subject_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(subject_rows[0]))
            writer.writeheader()
            writer.writerows(subject_rows)
    payload = {"task": "dep_hc", "config": dict(config), "metrics": result["metrics"]}
    (out_dir / "dep_hc_task_diagnostic.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "dep_hc_task_diagnostic.md").write_text(render_dep_hc_task_markdown(payload), encoding="utf-8")


def render_dep_hc_task_markdown(payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics", {})
    lines = [
        "# DEP/HC Task Diagnostic",
        "",
        "This is an independent DEP/HC cohort-classification task, not an emotion Top-4 route.",
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
