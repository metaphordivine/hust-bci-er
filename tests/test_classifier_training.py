from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import hust_bci_er.training.classifier as classifier_module
import hust_bci_er.training.reproducibility as reproducibility_module
from hust_bci_er.training.classifier import (
    ClassifierTrainConfig,
    EarlyStoppingConfig,
    EpochMetrics,
    OptimizerConfig,
    build_optimizer,
    clone_state_dict,
    classifier_train_config_from_route,
    evaluate_classifier,
    fit_classifier,
    logits_from_output,
    save_training_checkpoint,
    train_one_epoch,
)


@pytest.fixture(autouse=True)
def _lock_pythonhashseed(monkeypatch, request):
    seeds = {
        "test_fit_classifier_reduces_training_loss_for_tensor_batches": "0",
        "test_fit_classifier_accepts_mapping_batches_and_dict_model_output": "1",
        "test_fit_classifier_seed_resets_prebuilt_model_parameters": "7",
        "test_fit_classifier_can_preserve_preloaded_parameters": "11",
        "test_fit_classifier_ignores_checkpoint_when_train_config_differs": "0",
        "test_fit_classifier_ignores_checkpoint_when_resume_context_differs": "0",
        "test_fit_classifier_checkpoint_records_early_stop_before_resume": "0",
    }
    seed = seeds.get(request.node.name)
    if seed is not None:
        monkeypatch.setenv("PYTHONHASHSEED", seed)
        monkeypatch.setattr(reproducibility_module, "PROCESS_START_PYTHONHASHSEED", seed)


class DictOutputClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))

    def forward(self, x):
        return {"logits": self.net(x)}


class NoResetClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))

    def reset_parameters(self) -> None:
        raise AssertionError("preloaded model parameters must not be reset")

    def forward(self, x):
        return self.net(x)


def make_easy_loader(batch_size: int = 8, as_mapping: bool = False):
    torch.manual_seed(0)
    x = torch.randn(64, 2, 2)
    y = (x[:, 0, 0] + x[:, 1, 1] > 0).long()
    if not as_mapping:
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

    class MappingDataset(torch.utils.data.Dataset):
        def __len__(self):
            return int(x.shape[0])

        def __getitem__(self, idx):
            return {"features": x[idx], "target": y[idx]}

    return DataLoader(MappingDataset(), batch_size=batch_size, shuffle=False)


def test_fit_classifier_reduces_training_loss_for_tensor_batches():
    loader = make_easy_loader()
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    criterion = nn.CrossEntropyLoss()
    before = evaluate_classifier(model, loader, criterion=criterion)["loss"]

    result = fit_classifier(
        model,
        loader,
        config=ClassifierTrainConfig(
            epochs=12,
            seed=0,
            optimizer=OptimizerConfig(name="sgd", lr=0.2, momentum=0.0),
            early_stopping=EarlyStoppingConfig(monitor="train_loss", mode="min", patience=20),
        ),
        criterion=criterion,
    )

    after = evaluate_classifier(model, loader, criterion=criterion)["loss"]
    assert len(result.history) == 12
    assert after < before


def test_fit_classifier_can_preserve_preloaded_parameters():
    loader = make_easy_loader(batch_size=16)
    model = NoResetClassifier()

    result = fit_classifier(
        model,
        loader,
        config=ClassifierTrainConfig(
            epochs=1,
            seed=11,
            optimizer=OptimizerConfig(name="sgd", lr=0.01, momentum=0.0),
            early_stopping=EarlyStoppingConfig(monitor="train_loss", mode="min", patience=1),
            reset_parameters_after_seed=False,
        ),
    )

    assert result.best_epoch == 1
    assert result.best_state_dict is not None


def test_fit_classifier_accepts_mapping_batches_and_dict_model_output():
    loader = make_easy_loader(as_mapping=True)
    model = DictOutputClassifier()

    result = fit_classifier(
        model,
        loader,
        config=ClassifierTrainConfig(epochs=2, seed=1, optimizer=OptimizerConfig(name="adamw", lr=0.01)),
    )

    assert len(result.history) == 2
    assert result.history[-1].n_train == 64
    assert tuple(logits_from_output(model(torch.randn(3, 2, 2))).shape) == (3, 2)


def test_fit_classifier_seed_resets_prebuilt_model_parameters():
    loader = make_easy_loader()
    torch.manual_seed(123)
    model_a = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    torch.manual_seed(999)
    model_b = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))

    config = ClassifierTrainConfig(epochs=1, seed=7, optimizer=OptimizerConfig(name="sgd", lr=0.01, momentum=0.0))
    fit_classifier(model_a, loader, config=config)
    fit_classifier(model_b, loader, config=config)

    for left, right in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(left, right)


def test_fit_classifier_resumes_from_epoch_checkpoint(tmp_path):
    loader = make_easy_loader()
    config = ClassifierTrainConfig(
        epochs=3,
        seed=0,
        optimizer=OptimizerConfig(name="sgd", lr=0.05, momentum=0.0),
        early_stopping=EarlyStoppingConfig(monitor="train_loss", mode="min", patience=20),
    )
    checkpoint_path = tmp_path / "training_checkpoint.pt"
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    torch.manual_seed(0)
    optimizer = build_optimizer(model.parameters(), config.optimizer)
    criterion = nn.CrossEntropyLoss()
    first_epoch = train_one_epoch(
        model,
        loader,
        optimizer=optimizer,
        criterion=criterion,
        device=config.device,
        grad_clip_norm=config.grad_clip_norm,
    )
    first_metrics = EpochMetrics(
        epoch=1,
        train_loss=float(first_epoch["loss"]),
        train_accuracy=float(first_epoch["accuracy"]),
        n_train=int(first_epoch["n"]),
    )
    best_state = clone_state_dict(model)
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=1,
        history=[first_metrics],
        best_epoch=1,
        best_metric=first_metrics.train_loss,
        best_state_dict=best_state,
        stale_epochs=0,
        stopped_early=False,
        status="running",
    )

    resumed_model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    result = fit_classifier(resumed_model, loader, config=config, checkpoint_path=checkpoint_path)

    assert result.resumed_from_checkpoint is True
    assert [item.epoch for item in result.history] == [1, 2, 3]
    assert result.checkpoint_epoch == 3
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["status"] == "completed"


def test_fit_classifier_ignores_checkpoint_when_train_config_differs(tmp_path):
    loader = make_easy_loader()
    checkpoint_path = tmp_path / "training_checkpoint.pt"
    smoke_config = ClassifierTrainConfig(
        epochs=1,
        seed=0,
        optimizer=OptimizerConfig(name="sgd", lr=0.05, momentum=0.0),
        early_stopping=EarlyStoppingConfig(monitor="train_loss", mode="min", patience=20),
    )
    candidate_config = ClassifierTrainConfig(
        epochs=2,
        seed=0,
        optimizer=OptimizerConfig(name="sgd", lr=0.05, momentum=0.0),
        early_stopping=EarlyStoppingConfig(monitor="train_loss", mode="min", patience=20),
    )
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    optimizer = build_optimizer(model.parameters(), smoke_config.optimizer)
    metrics = EpochMetrics(epoch=1, train_loss=0.5, train_accuracy=0.75, n_train=64)
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=smoke_config,
        epoch=1,
        history=[metrics],
        best_epoch=1,
        best_metric=metrics.train_loss,
        best_state_dict=clone_state_dict(model),
        stale_epochs=0,
        stopped_early=False,
        status="running",
    )

    fresh_model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    result = fit_classifier(fresh_model, loader, config=candidate_config, checkpoint_path=checkpoint_path)

    assert result.resumed_from_checkpoint is False
    assert [item.epoch for item in result.history] == [1, 2]


def test_fit_classifier_ignores_checkpoint_when_resume_context_differs(tmp_path):
    loader = make_easy_loader()
    checkpoint_path = tmp_path / "training_checkpoint.pt"
    config = ClassifierTrainConfig(
        epochs=2,
        seed=0,
        optimizer=OptimizerConfig(name="sgd", lr=0.05, momentum=0.0),
        early_stopping=EarlyStoppingConfig(monitor="train_loss", mode="min", patience=20),
    )
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    optimizer = build_optimizer(model.parameters(), config.optimizer)
    metrics = EpochMetrics(epoch=1, train_loss=0.5, train_accuracy=0.75, n_train=64)
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=1,
        history=[metrics],
        best_epoch=1,
        best_metric=metrics.train_loss,
        best_state_dict=clone_state_dict(model),
        stale_epochs=0,
        stopped_early=False,
        status="running",
        resume_context={"route_id": "old", "split_id": "s1"},
    )

    fresh_model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    result = fit_classifier(
        fresh_model,
        loader,
        config=config,
        checkpoint_path=checkpoint_path,
        resume_context={"route_id": "new", "split_id": "s1"},
    )

    assert result.resumed_from_checkpoint is False
    assert [item.epoch for item in result.history] == [1, 2]


def test_fit_classifier_checkpoint_records_early_stop_before_resume(tmp_path):
    loader = make_easy_loader()
    checkpoint_path = tmp_path / "training_checkpoint.pt"
    config = ClassifierTrainConfig(
        epochs=5,
        seed=0,
        optimizer=OptimizerConfig(name="sgd", lr=0.01, momentum=0.0),
        early_stopping=EarlyStoppingConfig(monitor="train_loss", mode="min", patience=1, min_delta=1e9),
    )
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))

    result = fit_classifier(model, loader, config=config, checkpoint_path=checkpoint_path)

    assert result.stopped_early is True
    assert result.checkpoint_epoch == 2
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["stopped_early"] is True
    assert payload["stale_epochs"] == 1
    assert payload["status"] == "completed"


def test_save_training_checkpoint_retries_windows_replace_lock(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "training_checkpoint.pt"
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    config = ClassifierTrainConfig(epochs=1, seed=0, optimizer=OptimizerConfig(name="sgd", lr=0.01, momentum=0.0))
    optimizer = build_optimizer(model.parameters(), config.optimizer)
    metrics = EpochMetrics(epoch=1, train_loss=0.5, train_accuracy=0.75, n_train=4)
    path_type = type(checkpoint_path)
    original_replace = path_type.replace
    attempts: list[str] = []

    def flaky_replace(self: Path, target: Path) -> Path:
        if Path(target) == checkpoint_path and not attempts:
            attempts.append(self.name)
            raise PermissionError("simulated Windows file lock")
        return original_replace(self, target)

    monkeypatch.setattr(classifier_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(path_type, "replace", flaky_replace)

    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=1,
        history=[metrics],
        best_epoch=1,
        best_metric=0.5,
        best_state_dict=clone_state_dict(model),
        stale_epochs=0,
        stopped_early=False,
        status="running",
    )

    assert attempts
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_fit_classifier_rejects_val_monitor_without_val_loader():
    loader = make_easy_loader()
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))

    with pytest.raises(ValueError, match="requires val_loader"):
        fit_classifier(
            model,
            loader,
            config=ClassifierTrainConfig(
                epochs=2,
                optimizer=OptimizerConfig(name="sgd", lr=0.01, momentum=0.0),
                early_stopping=EarlyStoppingConfig(monitor="val_loss", mode="min", patience=1),
            ),
        )


def test_build_optimizer_rejects_unknown_name():
    model = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="unsupported optimizer"):
        build_optimizer(model.parameters(), OptimizerConfig(name="rmsprop"))


def test_classifier_train_config_from_route_uses_route_values():
    route_data = {
        "seed": 42,
        "training": {
            "trainer": "torch_classifier",
            "epochs": 5,
            "batch_size": 16,
            "optimizer": {"name": "adamw", "lr": 0.003, "weight_decay": 0.02},
            "loss": "cross_entropy",
            "grad_clip_norm": 1.5,
            "early_stopping": {"monitor": "val_accuracy", "mode": "max", "patience": 3, "min_delta": 0.01},
        },
    }

    config = classifier_train_config_from_route(route_data)

    assert config.seed == 42
    assert config.epochs == 5
    assert config.batch_size == 16
    assert config.optimizer.lr == 0.003
    assert config.early_stopping is not None
    assert config.early_stopping.monitor == "val_accuracy"
