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
    subject_threshold_diagnostic,
)
from hust_bci_er.tasks.dep_hc.experiment import dep_hc_prediction_rows


DEP_HC_NEURAL_MODELS = frozenset(
    {
        "eegnet",
        "shallow_conv_net",
        "conformer_lite",
        "deformer_lite",
        "fbstcnet",
        "srfnet",
        "tsception",
        "dgcnn",
        "lggnet",
        "dual_graph_conformer",
    }
)

_HUST_MONTAGE_MODELS = {"tsception", "dgcnn", "lggnet", "dual_graph_conformer"}

_DIAGNOSTIC_MODEL_KWARGS: dict[str, dict[str, Any]] = {
    "conformer_lite": {"emb_size": 32, "depth": 1, "num_heads": 4, "dropout": 0.4, "conv_dropout": 0.25},
    "deformer_lite": {"conv_channels": 16, "transformer_depth": 1, "embedding_dim": 32, "dropout": 0.35},
    "fbstcnet": {"variant": "M", "n_bands": 6, "F1": 18, "F2": 18, "gamma": 200, "pool_size": 80},
    "srfnet": {"emb_size": 24, "depth": 1, "num_heads": 4, "feature_dim": 48, "dropout": 0.35},
    "tsception": {"n_filters": 4, "temporal_kernel_sizes": (31, 63), "classifier_hidden_dim": 32, "dropout": 0.35},
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
    for _epoch in range(int(epochs)):
        order = torch.randperm(x_train.shape[0], generator=generator)
        model.train()
        for start in range(0, int(order.numel()), int(batch_size)):
            idx = order[start : start + int(batch_size)]
            xb = x_train[idx].to(torch_device)
            yb = y_train[idx].to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(_output_logits(model(xb)), yb)
            loss.backward()
            optimizer.step()

    val_p_dep = _predict_dep_probabilities(model, x_val, device=torch_device, batch_size=batch_size)
    threshold_summary = subject_threshold_diagnostic(
        val_samples,
        y_val.numpy(),
        val_p_dep,
        objective=threshold_objective,
        aggregation=subject_aggregation,
    )
    threshold = float(threshold_summary["threshold"])
    eval_p_dep = _predict_dep_probabilities(model, x_eval, device=torch_device, batch_size=batch_size)
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
        "subject_brier": brier_score(subject_truth, np.array([float(row["subject_score_p_dep"]) for row in subject_rows])),
        "n_train_windows": int(x_train.shape[0]),
        "n_eval_windows": int(x_eval.shape[0]),
        "n_eval_subjects": len(subject_rows),
        "threshold": threshold,
        "threshold_source": "fixed_0.5" if threshold_objective == "fixed_0_5" else "validation_subjects",
        "threshold_objective": threshold_objective,
        "class_weight_mode": class_weight_mode,
        "subject_aggregation": subject_aggregation,
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
    return {"model": model, "metrics": metrics, "prediction_rows": prediction_rows, "subject_rows": subject_rows}


def _build_model_kwargs(model_name: str, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    kwargs = dict(_DIAGNOSTIC_MODEL_KWARGS.get(str(model_name), {}))
    if model_name in _HUST_MONTAGE_MODELS:
        kwargs.setdefault("channel_montage", "hust_30_a2")
    if overrides:
        kwargs.update(dict(overrides))
    return kwargs


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


def _seed_torch(seed: int) -> None:
    _require_torch()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("DEP/HC neural diagnostics require torch")
