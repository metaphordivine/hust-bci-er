"""Reusable PyTorch classifier training loop."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
    reset_parameters_after_seed: bool = True


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
    resumed_from_checkpoint: bool = False
    checkpoint_path: str | None = None
    checkpoint_epoch: int = 0


def set_torch_seed(seed: int | None) -> None:
    if seed is None:
        return
    apply_reproducibility(ReproducibilityConfig(seed=int(seed)))


def reset_model_parameters(model: nn.Module) -> None:
    for module in model.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()


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
    checkpoint_path: Path | str | None = None,
    resume: bool = True,
    epoch_callback: Callable[[EpochMetrics], None] | None = None,
) -> TrainResult:
    config = config or ClassifierTrainConfig()
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.grad_clip_norm is not None and config.grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive")

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
    start_epoch = 1
    resumed_from_checkpoint = False
    checkpoint_path_obj = Path(checkpoint_path) if checkpoint_path is not None else None

    monitor = config.early_stopping or EarlyStoppingConfig(monitor="val_loss" if val_loader is not None else "train_loss")
    if config.early_stopping is not None and val_loader is None and monitor.monitor in {"val_loss", "val_accuracy"}:
        raise ValueError(f"early_stopping monitor {monitor.monitor} requires val_loader")

    if resume and checkpoint_path_obj is not None and checkpoint_path_obj.exists():
        checkpoint = load_training_checkpoint(checkpoint_path_obj, config=config)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        restore_rng_state(checkpoint)
        history = [epoch_metrics_from_dict(row) for row in checkpoint.get("history", [])]
        best_state = checkpoint.get("best_state_dict")
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_metric = float(checkpoint.get("best_metric", float("inf")))
        stopped_early = bool(checkpoint.get("stopped_early", False))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        last_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = last_epoch + 1
        resumed_from_checkpoint = True
        if stopped_early or last_epoch >= config.epochs:
            if best_state is not None and config.restore_best:
                model.load_state_dict(best_state)
            save_training_checkpoint(
                checkpoint_path_obj,
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=last_epoch,
                history=history,
                best_epoch=best_epoch,
                best_metric=best_metric,
                best_state_dict=best_state,
                stale_epochs=stale_epochs,
                stopped_early=stopped_early,
                status="completed",
            )
            return TrainResult(
                tuple(history),
                best_epoch,
                best_metric,
                stopped_early,
                best_state,
                resumed_from_checkpoint=True,
                checkpoint_path=str(checkpoint_path_obj),
                checkpoint_epoch=last_epoch,
            )
    else:
        set_torch_seed(config.seed)
        if config.seed is not None and config.reset_parameters_after_seed:
            reset_model_parameters(model)
        model.to(device)
        optimizer = optimizer or build_optimizer(model.parameters(), config.optimizer)

    for epoch in range(start_epoch, config.epochs + 1):
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
        if epoch_callback is not None:
            epoch_callback(epoch_metrics)

        metric = monitored_value(epoch_metrics, monitor.monitor)
        if metric is None:
            raise ValueError(f"early_stopping monitor {monitor.monitor} is unavailable for this training run")
        if is_improvement(metric, best_metric, mode=monitor.mode, min_delta=monitor.min_delta):
            best_metric = metric
            best_epoch = epoch
            best_state = clone_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1

        if checkpoint_path_obj is not None:
            save_training_checkpoint(
                checkpoint_path_obj,
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                history=history,
                best_epoch=best_epoch,
                best_metric=best_metric,
                best_state_dict=best_state,
                stale_epochs=stale_epochs,
                stopped_early=stopped_early,
                status="running",
            )

        if config.early_stopping is not None and stale_epochs > 0 and stale_epochs >= monitor.patience:
            stopped_early = True
            break

    if best_state is not None and config.restore_best:
        model.load_state_dict(best_state)
    if best_epoch == 0:
        best_epoch = history[-1].epoch
        best_metric = monitored_value(history[-1], "val_loss" if val_loader is not None else "train_loss") or history[-1].train_loss
        best_state = clone_state_dict(model)

    if checkpoint_path_obj is not None:
        save_training_checkpoint(
            checkpoint_path_obj,
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=history[-1].epoch,
            history=history,
            best_epoch=best_epoch,
            best_metric=best_metric,
            best_state_dict=best_state,
            stale_epochs=stale_epochs,
            stopped_early=stopped_early,
            status="completed",
        )

    return TrainResult(
        tuple(history),
        best_epoch,
        best_metric,
        stopped_early,
        best_state,
        resumed_from_checkpoint=resumed_from_checkpoint,
        checkpoint_path=str(checkpoint_path_obj) if checkpoint_path_obj is not None else None,
        checkpoint_epoch=history[-1].epoch,
    )


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


def epoch_metrics_to_dict(metrics: EpochMetrics) -> dict[str, Any]:
    return {
        "epoch": int(metrics.epoch),
        "train_loss": float(metrics.train_loss),
        "train_accuracy": float(metrics.train_accuracy),
        "n_train": int(metrics.n_train),
        "val_loss": float(metrics.val_loss) if metrics.val_loss is not None else None,
        "val_accuracy": float(metrics.val_accuracy) if metrics.val_accuracy is not None else None,
        "n_val": int(metrics.n_val),
    }


def epoch_metrics_from_dict(payload: Mapping[str, Any]) -> EpochMetrics:
    return EpochMetrics(
        epoch=int(payload["epoch"]),
        train_loss=float(payload["train_loss"]),
        train_accuracy=float(payload["train_accuracy"]),
        n_train=int(payload["n_train"]),
        val_loss=float(payload["val_loss"]) if payload.get("val_loss") is not None else None,
        val_accuracy=float(payload["val_accuracy"]) if payload.get("val_accuracy") is not None else None,
        n_val=int(payload.get("n_val", 0)),
    )


def train_config_signature(config: ClassifierTrainConfig) -> dict[str, Any]:
    return {
        "epochs": int(config.epochs),
        "batch_size": int(config.batch_size),
        "seed": int(config.seed) if config.seed is not None else None,
        "optimizer": {
            "name": str(config.optimizer.name),
            "lr": float(config.optimizer.lr),
            "weight_decay": float(config.optimizer.weight_decay),
            "momentum": float(config.optimizer.momentum),
        },
        "early_stopping": (
            {
                "monitor": str(config.early_stopping.monitor),
                "mode": str(config.early_stopping.mode),
                "patience": int(config.early_stopping.patience),
                "min_delta": float(config.early_stopping.min_delta),
            }
            if config.early_stopping is not None
            else None
        ),
        "grad_clip_norm": float(config.grad_clip_norm) if config.grad_clip_norm is not None else None,
        "restore_best": bool(config.restore_best),
        "reset_parameters_after_seed": bool(config.reset_parameters_after_seed),
    }


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ClassifierTrainConfig,
    epoch: int,
    history: list[EpochMetrics],
    best_epoch: int,
    best_metric: float,
    best_state_dict: dict[str, torch.Tensor] | None,
    stale_epochs: int,
    stopped_early: bool,
    status: str,
) -> None:
    payload = {
        "checkpoint_schema_version": 1,
        "artifact_kind": "torch_classifier_training_resume",
        "status": status,
        "epoch": int(epoch),
        "config_signature": train_config_signature(config),
        "history": [epoch_metrics_to_dict(item) for item in history],
        "model_state_dict": clone_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "best_state_dict": best_state_dict,
        "stale_epochs": int(stale_epochs),
        "stopped_early": bool(stopped_early),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_training_checkpoint(path: Path, *, config: ClassifierTrainConfig) -> Mapping[str, Any]:
    # weights_only=False is required because resume checkpoints contain optimizer
    # state and RNG tensors written by this training loop. Only load local run-dir
    # checkpoints created by the repository itself.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"training checkpoint must be a mapping: {path}")
    if payload.get("artifact_kind") != "torch_classifier_training_resume":
        raise ValueError(f"unsupported training checkpoint artifact_kind: {path}")
    if payload.get("config_signature") != train_config_signature(config):
        raise ValueError("training checkpoint config does not match this run")
    return payload


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_rng_state(checkpoint: Mapping[str, Any]) -> None:
    torch_rng_state = checkpoint.get("torch_rng_state")
    if torch.is_tensor(torch_rng_state):
        torch.set_rng_state(torch_rng_state)
    cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_state_all)
