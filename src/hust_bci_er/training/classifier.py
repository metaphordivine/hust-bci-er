"""Reusable PyTorch classifier training loop."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from hust_bci_er.training.reproducibility import ReproducibilityConfig, apply_reproducibility


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9


@dataclass(frozen=True)
class EarlyStoppingConfig:
    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 10
    min_delta: float = 0.0


@dataclass(frozen=True)
class ClassifierTrainConfig:
    epochs: int = 1
    batch_size: int = 32
    device: str | torch.device = "cpu"
    seed: int | None = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    early_stopping: EarlyStoppingConfig | None = None
    grad_clip_norm: float | None = None
    restore_best: bool = True


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    n_train: int
    val_loss: float | None = None
    val_accuracy: float | None = None
    n_val: int = 0


@dataclass(frozen=True)
class TrainResult:
    history: tuple[EpochMetrics, ...]
    best_epoch: int
    best_metric: float
    stopped_early: bool
    best_state_dict: dict[str, torch.Tensor] | None = None


def set_torch_seed(seed: int | None) -> None:
    if seed is None:
        return
    apply_reproducibility(ReproducibilityConfig(seed=int(seed)))


def logits_from_output(output: Any) -> torch.Tensor:
    logits = output.get("logits") if isinstance(output, Mapping) else output
    if not torch.is_tensor(logits):
        raise TypeError("model output must be a tensor or a mapping containing logits")
    if logits.ndim != 2:
        raise ValueError("classifier logits must be shaped [batch, classes]")
    return logits


def batch_to_tensors(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, Mapping):
        x = first_present(batch, ("x", "features", "inputs"))
        y = first_present(batch, ("y", "target", "targets", "label", "labels"))
        if x is None or y is None:
            raise KeyError("mapping batch must include feature and target tensors")
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        x, y = batch[0], batch[1]
    else:
        raise TypeError("batch must be a mapping or a tuple/list of (features, targets)")
    return torch.as_tensor(x, device=device).float(), torch.as_tensor(y, device=device).long()


def first_present(batch: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in batch:
            return batch[key]
    return None


def build_optimizer(parameters: Iterable[torch.nn.Parameter], config: OptimizerConfig) -> torch.optim.Optimizer:
    if config.lr <= 0:
        raise ValueError("optimizer lr must be positive")
    if config.weight_decay < 0:
        raise ValueError("optimizer weight_decay must be non-negative")
    name = config.name.lower()
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=config.lr, weight_decay=config.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=config.lr, weight_decay=config.weight_decay, momentum=config.momentum)
    raise ValueError(f"unsupported optimizer: {config.name}")


def classifier_train_config_from_route(route_data: Mapping[str, Any], *, device: str | torch.device = "cpu") -> ClassifierTrainConfig:
    training = route_data.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("route has no training mapping")
    if training.get("trainer") != "torch_classifier":
        raise ValueError("route training.trainer must be torch_classifier")
    if training.get("loss") != "cross_entropy":
        raise ValueError("route training.loss must be cross_entropy")

    optimizer_data = training.get("optimizer")
    if not isinstance(optimizer_data, Mapping):
        raise ValueError("route training.optimizer must be a mapping")
    early_data = training.get("early_stopping")

    return ClassifierTrainConfig(
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        device=device,
        seed=int(route_data["seed"]) if "seed" in route_data else None,
        optimizer=OptimizerConfig(
            name=str(optimizer_data["name"]),
            lr=float(optimizer_data["lr"]),
            weight_decay=float(optimizer_data.get("weight_decay", 0.0)),
            momentum=float(optimizer_data.get("momentum", 0.9)),
        ),
        early_stopping=(
            EarlyStoppingConfig(
                monitor=str(early_data.get("monitor", "val_loss")),
                mode=str(early_data.get("mode", "min")),
                patience=int(early_data.get("patience", 10)),
                min_delta=float(early_data.get("min_delta", 0.0)),
            )
            if isinstance(early_data, Mapping)
            else None
        ),
        grad_clip_norm=float(training["grad_clip_norm"]) if "grad_clip_norm" in training else None,
    )


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Any],
    *,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str | torch.device = "cpu",
    grad_clip_norm: float | None = None,
) -> dict[str, float | int]:
    model.train()
    device_obj = torch.device(device)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch in loader:
        x, y = batch_to_tensors(batch, device_obj)
        optimizer.zero_grad(set_to_none=True)
        logits = logits_from_output(model(x))
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        n = int(y.numel())
        total_loss += float(loss.detach().item()) * n
        total_correct += int((logits.detach().argmax(dim=1) == y).sum().item())
        total_seen += n

    if total_seen == 0:
        raise ValueError("training loader produced no examples")
    return {"loss": total_loss / total_seen, "accuracy": total_correct / total_seen, "n": total_seen}


def evaluate_classifier(
    model: nn.Module,
    loader: Iterable[Any],
    *,
    criterion: nn.Module,
    device: str | torch.device = "cpu",
) -> dict[str, float | int]:
    model.eval()
    device_obj = torch.device(device)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for batch in loader:
            x, y = batch_to_tensors(batch, device_obj)
            logits = logits_from_output(model(x))
            loss = criterion(logits, y)

            n = int(y.numel())
            total_loss += float(loss.item()) * n
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total_seen += n

    if total_seen == 0:
        raise ValueError("evaluation loader produced no examples")
    return {"loss": total_loss / total_seen, "accuracy": total_correct / total_seen, "n": total_seen}


def fit_classifier(
    model: nn.Module,
    train_loader: Iterable[Any],
    *,
    val_loader: Iterable[Any] | None = None,
    config: ClassifierTrainConfig | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    criterion: nn.Module | None = None,
) -> TrainResult:
    config = config or ClassifierTrainConfig()
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.grad_clip_norm is not None and config.grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive")

    set_torch_seed(config.seed)
    device = torch.device(config.device)
    model.to(device)
    criterion = criterion or nn.CrossEntropyLoss()
    optimizer = optimizer or build_optimizer(model.parameters(), config.optimizer)

    history: list[EpochMetrics] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_metric = float("inf")
    stopped_early = False
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
        )
        val_metrics = evaluate_classifier(model, val_loader, criterion=criterion, device=device) if val_loader is not None else None
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=float(train_metrics["loss"]),
            train_accuracy=float(train_metrics["accuracy"]),
            n_train=int(train_metrics["n"]),
            val_loss=float(val_metrics["loss"]) if val_metrics else None,
            val_accuracy=float(val_metrics["accuracy"]) if val_metrics else None,
            n_val=int(val_metrics["n"]) if val_metrics else 0,
        )
        history.append(epoch_metrics)

        monitor = config.early_stopping or EarlyStoppingConfig(monitor="val_loss" if val_loader is not None else "train_loss")
        metric = monitored_value(epoch_metrics, monitor.monitor)
        if metric is None:
            continue
        if is_improvement(metric, best_metric, mode=monitor.mode, min_delta=monitor.min_delta):
            best_metric = metric
            best_epoch = epoch
            best_state = clone_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1

        if config.early_stopping is not None and stale_epochs > monitor.patience:
            stopped_early = True
            break

    if best_state is not None and config.restore_best:
        model.load_state_dict(best_state)
    if best_epoch == 0:
        best_epoch = history[-1].epoch
        best_metric = monitored_value(history[-1], "val_loss" if val_loader is not None else "train_loss") or history[-1].train_loss
        best_state = clone_state_dict(model)

    return TrainResult(tuple(history), best_epoch, best_metric, stopped_early, best_state)


def monitored_value(metrics: EpochMetrics, name: str) -> float | None:
    if name == "train_loss":
        return metrics.train_loss
    if name == "train_accuracy":
        return metrics.train_accuracy
    if name == "val_loss":
        return metrics.val_loss
    if name == "val_accuracy":
        return metrics.val_accuracy
    raise ValueError(f"unsupported monitor: {name}")


def is_improvement(value: float, best: float, *, mode: str, min_delta: float = 0.0) -> bool:
    if mode == "min":
        return value < best - min_delta
    if mode == "max":
        if best == float("inf"):
            return True
        return value > best + min_delta
    raise ValueError(f"unsupported monitor mode: {mode}")


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
