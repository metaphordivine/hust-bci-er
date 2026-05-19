from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn
from torch.utils.data import DataLoader

from hust_bci_er.adaptation.dann import DomainAdversarialClassifier
from hust_bci_er.training.classifier import (
    ClassifierTrainConfig,
    EarlyStoppingConfig,
    OptimizerConfig,
    coral_alignment_loss,
    fit_domain_coral_classifier,
    fit_domain_adversarial_classifier,
    fit_masked_consistency_classifier,
    masked_feature_consistency_loss,
    masked_time_batch,
    train_one_domain_adversarial_epoch,
)


class TinyFeatureClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(4, 6), nn.GELU())
        self.classifier = nn.Linear(6, 2)

    def extract_features(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.classifier(self.extract_features(x))


class DomainDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(24, 2, 2, generator=g)
        self.y = (self.x[:, 0, 0] > 0).long()
        self.domain = (self.x[:, 1, 1] > 0).long()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return {"features": self.x[idx], "target": self.y[idx], "domain": self.domain[idx]}


class FixedDomainAdversarialModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.seen_lambdas: list[float] = []

    def extract_features(self, x):
        return torch.ones(x.shape[0], 2, device=x.device) + self.anchor * 0

    def label_logits_from_features(self, features):
        return torch.zeros(features.shape[0], 2, device=features.device) + self.anchor * 0

    def domain_logits_from_features(self, features, lambd: float = 1.0):
        self.seen_lambdas.append(float(lambd))
        return torch.zeros(features.shape[0], 2, device=features.device) + self.anchor * 0


def test_domain_adversarial_classifier_shapes():
    model = DomainAdversarialClassifier(TinyFeatureClassifier(), feature_dim=6, domain_hidden_dim=4)
    x = torch.randn(3, 2, 2)
    features = model.extract_features(x)

    assert tuple(model(x).shape) == (3, 2)
    assert tuple(model.domain_logits_from_features(features, lambd=0.5).shape) == (3, 2)


def test_domain_lambda_is_applied_once_through_grl():
    model = FixedDomainAdversarialModel()
    batch = {
        "features": torch.zeros(2, 2, 2),
        "target": torch.tensor([0, 1]),
        "domain": torch.tensor([0, 1]),
    }

    metrics = train_one_domain_adversarial_epoch(
        model,
        [batch],
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        criterion=nn.CrossEntropyLoss(),
        domain_criterion=nn.CrossEntropyLoss(),
        domain_lambda=0.05,
    )

    assert model.seen_lambdas == [pytest.approx(0.05)]
    assert metrics["loss"] == pytest.approx(metrics["label_loss"] + metrics["domain_loss"])


def test_fit_domain_adversarial_classifier_writes_resume_checkpoint(tmp_path):
    model = DomainAdversarialClassifier(TinyFeatureClassifier(), feature_dim=6, domain_hidden_dim=4)
    loader = DataLoader(DomainDataset(), batch_size=6, shuffle=False)
    checkpoint_path = tmp_path / "dann_training_checkpoint.pt"

    result = fit_domain_adversarial_classifier(
        model,
        loader,
        config=ClassifierTrainConfig(
            epochs=2,
            seed=None,
            optimizer=OptimizerConfig(name="sgd", lr=0.02, momentum=0.0),
            early_stopping=EarlyStoppingConfig(monitor="train_loss", mode="min", patience=5),
        ),
        domain_lambda=0.05,
        checkpoint_path=checkpoint_path,
        resume_context={"route_id": "tiny_dann"},
    )

    assert len(result.history) == 2
    assert checkpoint_path.exists()


def test_coral_checkpoint_rejects_alignment_lambda_change(tmp_path):
    loader = DataLoader(DomainDataset(), batch_size=6, shuffle=False)
    checkpoint_path = tmp_path / "coral_lambda_signature.pt"
    config = ClassifierTrainConfig(
        epochs=1,
        seed=None,
        optimizer=OptimizerConfig(name="sgd", lr=0.02, momentum=0.0),
        early_stopping=EarlyStoppingConfig(monitor="train_loss", mode="min", patience=5),
    )

    fit_domain_coral_classifier(
        TinyFeatureClassifier(),
        loader,
        config=config,
        alignment_lambda=0.03,
        checkpoint_path=checkpoint_path,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["resume_context_signature"]["training_objective"]["alignment_lambda"] == pytest.approx(0.03)

    result = fit_domain_coral_classifier(
        TinyFeatureClassifier(),
        loader,
        config=config,
        alignment_lambda=0.3,
        checkpoint_path=checkpoint_path,
    )

    assert result.resumed_from_checkpoint is False


def test_domain_adversarial_checkpoint_rejects_domain_lambda_change(tmp_path):
    loader = DataLoader(DomainDataset(), batch_size=6, shuffle=False)
    checkpoint_path = tmp_path / "dann_lambda_signature.pt"
    config = ClassifierTrainConfig(
        epochs=1,
        seed=None,
        optimizer=OptimizerConfig(name="sgd", lr=0.02, momentum=0.0),
        early_stopping=EarlyStoppingConfig(monitor="train_loss", mode="min", patience=5),
    )

    fit_domain_adversarial_classifier(
        DomainAdversarialClassifier(TinyFeatureClassifier(), feature_dim=6, domain_hidden_dim=4),
        loader,
        config=config,
        domain_lambda=0.05,
        checkpoint_path=checkpoint_path,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["resume_context_signature"]["training_objective"]["domain_lambda"] == pytest.approx(0.05)

    result = fit_domain_adversarial_classifier(
        DomainAdversarialClassifier(TinyFeatureClassifier(), feature_dim=6, domain_hidden_dim=4),
        loader,
        config=config,
        domain_lambda=1.0,
        checkpoint_path=checkpoint_path,
    )

    assert result.resumed_from_checkpoint is False


def test_domain_adversarial_checkpoint_records_early_stop_before_resume(tmp_path):
    model = DomainAdversarialClassifier(TinyFeatureClassifier(), feature_dim=6, domain_hidden_dim=4)
    loader = DataLoader(DomainDataset(), batch_size=6, shuffle=False)
    checkpoint_path = tmp_path / "dann_early_stop.pt"

    result = fit_domain_adversarial_classifier(
        model,
        loader,
        config=ClassifierTrainConfig(
            epochs=5,
            seed=None,
            optimizer=OptimizerConfig(name="sgd", lr=0.02, momentum=0.0),
            early_stopping=EarlyStoppingConfig(monitor="train_loss", mode="min", patience=1, min_delta=1e9),
        ),
        domain_lambda=0.05,
        checkpoint_path=checkpoint_path,
        resume_context={"route_id": "tiny_dann_stop"},
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert result.stopped_early is True
    assert result.checkpoint_epoch == 2
    assert payload["stopped_early"] is True


def test_coral_alignment_loss_and_training_checkpoint(tmp_path):
    features = torch.tensor(
        [
            [1.0, 0.0],
            [1.2, 0.1],
            [-1.0, 0.0],
            [-1.1, -0.2],
        ]
    )
    domains = torch.tensor([0, 0, 1, 1])
    assert float(coral_alignment_loss(features, domains)) > 0.0

    model = TinyFeatureClassifier()
    loader = DataLoader(DomainDataset(), batch_size=6, shuffle=False)
    checkpoint_path = tmp_path / "coral_training_checkpoint.pt"
    result = fit_domain_coral_classifier(
        model,
        loader,
        config=ClassifierTrainConfig(
            epochs=2,
            seed=None,
            optimizer=OptimizerConfig(name="sgd", lr=0.02, momentum=0.0),
            early_stopping=EarlyStoppingConfig(monitor="train_loss", mode="min", patience=5),
        ),
        alignment_lambda=0.05,
        checkpoint_path=checkpoint_path,
        resume_context={"route_id": "tiny_coral"},
    )

    assert len(result.history) == 2
    assert checkpoint_path.exists()


def test_masked_consistency_loss_and_training_checkpoint(tmp_path):
    x = torch.ones(3, 2, 8)
    masked = masked_time_batch(x, max_mask_width_samples=2)
    assert tuple(masked.shape) == tuple(x.shape)
    assert torch.any(masked != x)

    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    same = reference.clone()
    shifted = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert float(masked_feature_consistency_loss(reference, same)) == pytest.approx(0.0)
    assert float(masked_feature_consistency_loss(reference, shifted)) > 0.0

    model = TinyFeatureClassifier()
    loader = DataLoader(DomainDataset(), batch_size=6, shuffle=False)
    checkpoint_path = tmp_path / "masked_consistency_checkpoint.pt"
    result = fit_masked_consistency_classifier(
        model,
        loader,
        config=ClassifierTrainConfig(
            epochs=2,
            seed=None,
            optimizer=OptimizerConfig(name="sgd", lr=0.02, momentum=0.0),
            early_stopping=EarlyStoppingConfig(monitor="train_loss", mode="min", patience=5),
        ),
        consistency_lambda=0.05,
        max_mask_width_samples=1,
        checkpoint_path=checkpoint_path,
        resume_context={"route_id": "tiny_masked_consistency"},
    )

    assert len(result.history) == 2
    assert checkpoint_path.exists()
