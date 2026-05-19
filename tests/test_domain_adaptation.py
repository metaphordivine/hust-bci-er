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
    fit_domain_adversarial_classifier,
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


def test_domain_adversarial_classifier_shapes():
    model = DomainAdversarialClassifier(TinyFeatureClassifier(), feature_dim=6, domain_hidden_dim=4)
    x = torch.randn(3, 2, 2)
    features = model.extract_features(x)

    assert tuple(model(x).shape) == (3, 2)
    assert tuple(model.domain_logits_from_features(features, lambd=0.5).shape) == (3, 2)


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
