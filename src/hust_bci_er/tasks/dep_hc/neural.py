"""Small neural sanity baselines for independent DEP/HC classification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:  # Torch is optional in the lightweight CI environment.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised by CI import collection.
    torch = None
    nn = None

from hust_bci_er.analysis.dep_hc_router import (
    RouterSample,
    aggregate_subject_rows,
    balanced_accuracy_binary,
    brier_score,
    cohort_label,
    crop_combo_subject_metrics,
    infer_crop_combo_shape,
    subject_threshold_diagnostic,
)
from hust_bci_er.tasks.dep_hc.experiment import dep_hc_prediction_rows


DEP_HC_NEURAL_MODELS = frozenset(
    {
        "eegnet",
        "shallow_conv_net",
        "conformer_lite",
        "deformer_lite",
        "cbramod",
        "fbcnet",
        "fbstcnet",
        "riemannian_tangent",
        "srfnet",
        "srf_fbstcnet_gate",
        "tsception",
        "tri_context_gate",
        "dgcnn",
        "lggnet",
        "dual_graph_conformer",
    }
)

_HUST_MONTAGE_MODELS = {"tsception", "dgcnn", "lggnet", "dual_graph_conformer"}

_DIAGNOSTIC_MODEL_KWARGS: dict[str, dict[str, Any]] = {
    "conformer_lite": {"emb_size": 32, "depth": 1, "num_heads": 4, "dropout": 0.4, "conv_dropout": 0.25},
    "deformer_lite": {"conv_channels": 16, "transformer_depth": 1, "embedding_dim": 32, "dropout": 0.35},
    "cbramod": {
        "patch_size": 200,
        "d_model": 64,
        "emb_dim": 64,
        "n_layer": 2,
        "nhead": 4,
        "dim_feedforward": 128,
        "drop_prob": 0.2,
        "classifier_pooling": "mean",
        "conv_out_channels": 8,
        "group_norm_groups": 4,
    },
    "fbcnet": {
        "n_bands": 6,
        "spatial_filters": 3,
        "temporal_kernel_size": 63,
        "n_segments": 5,
        "classifier_hidden_dim": 32,
        "dropout": 0.4,
    },
    "fbstcnet": {"variant": "M", "n_bands": 6, "F1": 18, "F2": 18, "gamma": 200, "pool_size": 80},
    "riemannian_tangent": {"covariance_eps": 1e-3, "shrinkage": 0.1, "feature_standardize": True},
    "srfnet": {"emb_size": 24, "depth": 1, "num_heads": 4, "feature_dim": 48, "dropout": 0.35},
    "srf_fbstcnet_gate": {
        "srf": {"emb_size": 16, "depth": 1, "num_heads": 4, "feature_dim": 32, "dropout": 0.35},
        "fbstcnet": {"variant": "M", "n_bands": 4, "F1": 8, "F2": 8, "gamma": 100, "pool_size": 80},
        "gate_hidden_dim": 12,
        "gate_dropout": 0.2,
        "fusion_temperature": 1.0,
    },
    "tsception": {"n_filters": 4, "temporal_kernel_sizes": (31, 63), "classifier_hidden_dim": 32, "dropout": 0.35},
    "tri_context_gate": {
        "srf": {"emb_size": 16, "depth": 1, "num_heads": 4, "feature_dim": 32, "dropout": 0.35},
        "fbstcnet": {"variant": "M", "n_bands": 4, "F1": 8, "F2": 8, "gamma": 100, "pool_size": 80},
        "conformer": {"emb_size": 24, "depth": 1, "num_heads": 4, "dropout": 0.4, "conv_dropout": 0.25},
        "gate_hidden_dim": 16,
        "gate_dropout": 0.2,
        "fusion_temperature": 1.0,
    },
    "dgcnn": {
        "node_features": 8,
        "graph_hidden_dim": 8,
        "k_order": 2,
        "temporal_kernel_size": 31,
        "classifier_hidden_dim": 16,
        "dropout": 0.35,
    },
    "lggnet": {
        "temporal_filters": 4,
        "temporal_kernel_sizes": (15, 31),
        "graph_hidden_dim": 8,
        "classifier_hidden_dim": 16,
        "dropout": 0.35,
    },
    "dual_graph_conformer": {
        "embedding_dim": 16,
        "graph_hidden_dim": 8,
        "fusion_dim": 16,
        "k_order": 2,
        "num_heads": 4,
        "transformer_depth": 1,
        "temporal_kernel_size": 31,
        "token_count": 4,
        "classifier_hidden_dim": 16,
        "dropout": 0.35,
    },
}


def evaluate_dep_hc_neural_task(
    train_samples: Sequence[RouterSample],
    eval_samples: Sequence[RouterSample],
    *,
    val_samples: Sequence[RouterSample],
    model_name: str = "eegnet",
    device: str = "cpu",
    seed: int = 42,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    class_weight_mode: str = "balanced",
    threshold_objective: str = "balanced_accuracy",
    subject_aggregation: str = "mean",
    model_kwargs: Mapping[str, Any] | None = None,
    calibration_method: str = "none",
    sampling_strategy: str = "window",
) -> dict[str, Any]:
    _require_torch()
    if not train_samples or not eval_samples:
        raise ValueError("train_samples and eval_samples are required")
    if model_name not in DEP_HC_NEURAL_MODELS:
        raise ValueError(f"unknown DEP/HC neural model_name: {model_name}")
    if not val_samples:
        raise ValueError("neural DEP/HC diagnostics require validation subjects for threshold selection")
    if class_weight_mode not in {"balanced", "uniform"}:
        raise ValueError(f"unknown class_weight_mode: {class_weight_mode}")
    if calibration_method not in {"none", "temperature", "platt", "isotonic"}:
        raise ValueError(f"unknown calibration_method: {calibration_method}")
    if sampling_strategy not in {"window", "subject_balanced"}:
        raise ValueError(f"unknown sampling_strategy: {sampling_strategy}")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    _seed_torch(seed)
    from hust_bci_er.models.factory import build_model

    x_train, y_train = _sample_tensors(train_samples)
    x_val, y_val = _sample_tensors(val_samples)
    x_eval, y_eval = _sample_tensors(eval_samples)
    build_kwargs = _build_model_kwargs(model_name, model_kwargs)
    model = build_model(
        model_name,
        n_channels=int(x_train.shape[1]),
        n_times=int(x_train.shape[2]),
        n_classes=2,
        **build_kwargs,
    )
    torch_device = torch.device(device)
    model.to(torch_device)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_train, class_weight_mode=class_weight_mode).to(torch_device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    generator = torch.Generator().manual_seed(int(seed))
    sample_weights = (
        _subject_balanced_sample_weights(train_samples, y_train)
        if sampling_strategy == "subject_balanced"
        else None
    )
    for _epoch in range(int(epochs)):
        if sample_weights is None:
            order = torch.randperm(x_train.shape[0], generator=generator)
        else:
            order = torch.multinomial(sample_weights, int(x_train.shape[0]), replacement=True, generator=generator)
        model.train()
        for start in range(0, int(order.numel()), int(batch_size)):
            idx = order[start : start + int(batch_size)]
            xb = x_train[idx].to(torch_device)
            yb = y_train[idx].to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(_output_logits(model(xb)), yb)
            loss.backward()
            optimizer.step()

    val_p_dep_raw = _predict_dep_probabilities(model, x_val, device=torch_device, batch_size=batch_size)
    calibrator = _fit_probability_calibrator(y_val.numpy(), val_p_dep_raw, method=calibration_method)
    val_p_dep = _apply_probability_calibrator(val_p_dep_raw, calibrator)
    threshold_summary = subject_threshold_diagnostic(
        val_samples,
        y_val.numpy(),
        val_p_dep,
        objective=threshold_objective,
        aggregation=subject_aggregation,
    )
    threshold = float(threshold_summary["threshold"])
    eval_p_dep_raw = _predict_dep_probabilities(model, x_eval, device=torch_device, batch_size=batch_size)
    eval_p_dep = _apply_probability_calibrator(eval_p_dep_raw, calibrator)
    y_pred = (eval_p_dep >= threshold).astype(int)
    prediction_rows = dep_hc_prediction_rows(eval_samples, y_eval.numpy(), eval_p_dep, y_pred)
    subject_rows = aggregate_subject_rows(prediction_rows, threshold=threshold, aggregation=subject_aggregation)
    subject_truth = np.array([cohort_label(row["cohort"]) for row in subject_rows], dtype=int)
    subject_pred = np.array([cohort_label(row["predicted_cohort"]) for row in subject_rows], dtype=int)
    metrics: dict[str, Any] = {
        "task": "dep_hc",
        "model_name": model_name,
        "window_ba": balanced_accuracy_binary(y_eval.numpy(), y_pred),
        "subject_ba": balanced_accuracy_binary(subject_truth, subject_pred),
        "window_brier": brier_score(y_eval.numpy(), eval_p_dep),
        "window_brier_raw": brier_score(y_eval.numpy(), eval_p_dep_raw),
        "validation_window_brier": brier_score(y_val.numpy(), val_p_dep),
        "validation_window_brier_raw": brier_score(y_val.numpy(), val_p_dep_raw),
        "subject_brier": brier_score(subject_truth, np.array([float(row["subject_score_p_dep"]) for row in subject_rows])),
        "n_train_windows": int(x_train.shape[0]),
        "n_eval_windows": int(x_eval.shape[0]),
        "n_eval_subjects": len(subject_rows),
        "threshold": threshold,
        "threshold_source": "fixed_0.5" if threshold_objective == "fixed_0_5" else "validation_subjects",
        "threshold_objective": threshold_objective,
        "class_weight_mode": class_weight_mode,
        "subject_aggregation": subject_aggregation,
        "calibration_method": calibration_method,
        "calibration_params": _jsonable_calibrator(calibrator),
        "sampling_strategy": sampling_strategy,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "model_kwargs": _jsonable_model_kwargs(build_kwargs),
    }
    for key, value in threshold_summary.items():
        if key in {"objective", "threshold"}:
            continue
        metrics[f"validation_threshold_{key}"] = value if isinstance(value, str) else float(value)
    for cohort, label in {"HC": 0, "DEP": 1}.items():
        mask = subject_truth == label
        metrics[f"{cohort.lower()}_subject_recall"] = float(np.mean(subject_pred[mask] == label)) if np.any(mask) else float("nan")
    n_trials, n_crops = infer_crop_combo_shape(prediction_rows)
    metrics.update(
        crop_combo_subject_metrics(
            prediction_rows,
            threshold=threshold,
            aggregation=subject_aggregation,
            n_trials=n_trials,
            n_crops=n_crops,
        )
    )
    return {"model": model, "metrics": metrics, "prediction_rows": prediction_rows, "subject_rows": subject_rows}


def _build_model_kwargs(model_name: str, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    kwargs = dict(_DIAGNOSTIC_MODEL_KWARGS.get(str(model_name), {}))
    explicit_keys: set[str] = set()
    if model_name in _HUST_MONTAGE_MODELS:
        kwargs.setdefault("channel_montage", "hust_30_a2")
    if overrides:
        override_dict = dict(overrides)
        explicit_keys = {str(key) for key in override_dict}
        kwargs.update(override_dict)
    if model_name == "cbramod":
        _normalize_cbramod_patch_kwargs(kwargs, explicit_keys=explicit_keys)
    return kwargs


def _normalize_cbramod_patch_kwargs(kwargs: dict[str, Any], *, explicit_keys: set[str]) -> None:
    """Keep route-derived CBraMod patch settings internally consistent."""

    patch_size = int(kwargs.get("patch_size", 200))
    d_model = int(kwargs.get("d_model", 200))
    temporal_width = _cbramod_temporal_width(patch_size)
    if "conv_out_channels" not in explicit_keys:
        if d_model % temporal_width != 0:
            raise ValueError(
                "CBraMod route conversion cannot infer conv_out_channels because "
                f"d_model={d_model} is not divisible by temporal_width={temporal_width}"
            )
        kwargs["conv_out_channels"] = d_model // temporal_width
    else:
        conv_out_channels = int(kwargs["conv_out_channels"])
        if conv_out_channels * temporal_width != d_model:
            raise ValueError(
                "CBraMod patch embedding requires conv_out_channels * temporal_width "
                f"to equal d_model, got {conv_out_channels * temporal_width} vs {d_model}"
            )

    conv_out_channels = int(kwargs["conv_out_channels"])
    group_norm_groups = int(kwargs.get("group_norm_groups", 5))
    if conv_out_channels % group_norm_groups != 0:
        if "group_norm_groups" in explicit_keys:
            raise ValueError(
                "CBraMod requires conv_out_channels divisible by group_norm_groups, "
                f"got conv_out_channels={conv_out_channels} and group_norm_groups={group_norm_groups}"
            )
        kwargs["group_norm_groups"] = _preferred_group_norm_groups(conv_out_channels)


def _cbramod_temporal_width(patch_size: int) -> int:
    width = int(patch_size)
    for kernel_size, stride, padding in ((49, 25, 24), (3, 1, 1), (3, 1, 1)):
        width = ((width + 2 * padding - kernel_size) // stride) + 1
    return width


def _preferred_group_norm_groups(channels: int) -> int:
    for candidate in (5, 4, 3, 2, 1):
        if int(channels) % candidate == 0:
            return candidate
    return 1


def _jsonable_model_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def _output_logits(output: Any) -> torch.Tensor:
    _require_torch()
    if isinstance(output, dict):
        if "logits" not in output:
            raise KeyError("model output dictionary must contain logits")
        return output["logits"]
    return output


def _sample_tensors(samples: Sequence[RouterSample]) -> tuple[torch.Tensor, torch.Tensor]:
    _require_torch()
    x = np.stack([np.asarray(sample.x, dtype=np.float32) for sample in samples], axis=0)
    y = np.asarray([cohort_label(sample.cohort) for sample in samples], dtype=np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def _class_weights(labels: torch.Tensor, *, class_weight_mode: str) -> torch.Tensor:
    _require_torch()
    if class_weight_mode == "uniform":
        return torch.ones(2, dtype=torch.float32)
    counts = torch.bincount(labels, minlength=2).to(torch.float32)
    if torch.any(counts <= 0):
        raise ValueError("training labels must contain both DEP and HC")
    return labels.numel() / (2.0 * counts)


def _subject_balanced_sample_weights(samples: Sequence[RouterSample], labels: torch.Tensor) -> torch.Tensor:
    """Weight windows so each cohort and subject has equal expected epoch mass."""

    _require_torch()
    if len(samples) != int(labels.numel()):
        raise ValueError("samples and labels have incompatible lengths")
    labels_np = labels.cpu().numpy().astype(int)
    subject_keys = [str(sample.subject_id) for sample in samples]
    class_to_subjects: dict[int, set[str]] = {0: set(), 1: set()}
    subject_window_counts: dict[tuple[int, str], int] = {}
    for label, subject_id in zip(labels_np, subject_keys):
        class_to_subjects[int(label)].add(subject_id)
        subject_window_counts[(int(label), subject_id)] = subject_window_counts.get((int(label), subject_id), 0) + 1
    if not class_to_subjects[0] or not class_to_subjects[1]:
        raise ValueError("training labels must contain both DEP and HC")
    weights = []
    for label, subject_id in zip(labels_np, subject_keys):
        n_subjects = len(class_to_subjects[int(label)])
        n_windows = subject_window_counts[(int(label), subject_id)]
        weights.append(1.0 / (2.0 * float(n_subjects) * float(n_windows)))
    return torch.as_tensor(weights, dtype=torch.float64)


def _predict_dep_probabilities(
    model: Any,
    x: torch.Tensor,
    *,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    _require_torch()
    probs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            xb = x[start : start + int(batch_size)].to(device)
            prob = torch.softmax(_output_logits(model(xb)), dim=1)[:, 1]
            probs.append(prob.cpu().numpy())
    return np.concatenate(probs).astype(np.float64, copy=False)


def _fit_probability_calibrator(y_true: np.ndarray, p_dep: np.ndarray, *, method: str) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=np.float64).reshape(-1)
    probs = np.clip(np.asarray(p_dep, dtype=np.float64).reshape(-1), 1e-6, 1.0 - 1e-6)
    if labels.shape[0] != probs.shape[0]:
        raise ValueError("calibration labels and probabilities have incompatible lengths")
    if method == "none":
        return {"method": "none"}
    logits = _logit(probs)
    if method == "temperature":
        candidates = np.logspace(-1.0, 1.0, num=41)
        best_temp = 1.0
        best_key: tuple[float, float] | None = None
        for temp in candidates:
            calibrated = _sigmoid(logits / float(temp))
            key = (_binary_nll(labels, calibrated), abs(float(temp) - 1.0))
            if best_key is None or key < best_key:
                best_key = key
                best_temp = float(temp)
        return {"method": "temperature", "temperature": best_temp}
    if method == "platt":
        coef = np.array([1.0, 0.0], dtype=np.float64)
        lr = 0.05
        for _ in range(400):
            calibrated = _sigmoid(coef[0] * logits + coef[1])
            err = calibrated - labels
            coef[0] -= lr * float(np.mean(err * logits))
            coef[1] -= lr * float(np.mean(err))
        return {"method": "platt", "coef": float(coef[0]), "intercept": float(coef[1])}
    if method == "isotonic":
        try:
            from sklearn.isotonic import IsotonicRegression
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional local environment.
            raise ModuleNotFoundError("scikit-learn is required for isotonic DEP/HC calibration") from exc
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(probs, labels)
        return {
            "method": "isotonic",
            "x_thresholds": [float(value) for value in model.X_thresholds_],
            "y_thresholds": [float(value) for value in model.y_thresholds_],
        }
    raise ValueError(f"unknown calibration_method: {method}")


def _apply_probability_calibrator(p_dep: np.ndarray, calibrator: Mapping[str, Any]) -> np.ndarray:
    probs = np.clip(np.asarray(p_dep, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    method = str(calibrator.get("method", "none"))
    if method == "none":
        return probs
    logits = _logit(probs)
    if method == "temperature":
        temp = max(float(calibrator.get("temperature", 1.0)), 1e-6)
        return _sigmoid(logits / temp)
    if method == "platt":
        return _sigmoid(float(calibrator.get("coef", 1.0)) * logits + float(calibrator.get("intercept", 0.0)))
    if method == "isotonic":
        x = np.asarray(calibrator.get("x_thresholds", []), dtype=np.float64)
        y = np.asarray(calibrator.get("y_thresholds", []), dtype=np.float64)
        if x.size == 0 or y.size == 0:
            raise ValueError("isotonic calibrator is missing thresholds")
        return np.clip(np.interp(probs, x, y), 0.0, 1.0)
    raise ValueError(f"unknown calibration_method: {method}")


def _logit(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def _binary_nll(labels: np.ndarray, probs: np.ndarray) -> float:
    clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))


def _jsonable_calibrator(calibrator: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in calibrator.items():
        if isinstance(value, np.ndarray):
            out[key] = [float(item) for item in value.tolist()]
        elif isinstance(value, (list, tuple)):
            out[key] = [float(item) if isinstance(item, (np.floating, float, int)) else item for item in value]
        elif isinstance(value, (np.floating, float)):
            out[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            out[key] = int(value)
        else:
            out[key] = value
    return out


def _seed_torch(seed: int) -> None:
    _require_torch()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("DEP/HC neural diagnostics require torch")
