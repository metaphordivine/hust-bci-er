import pytest

torch = pytest.importorskip("torch")

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
