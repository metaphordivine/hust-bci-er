"""Central registry for valid route component names."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def route_statuses() -> set[str]:
    path = ROOT / "configs" / "statuses.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return set((data.get("statuses") or {}).keys())


@lru_cache(maxsize=1)
def audit_decisions() -> set[str]:
    path = ROOT / "configs" / "audit_decisions.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return set((data.get("decisions") or {}).keys())


ROUTE_STATUSES = route_statuses()
AUDIT_DECISIONS = audit_decisions()

PREPROCESSING = {
    "zscore",
    "euclidean_alignment",
    "whitening_eps1e3",
    "whitening_eps3e4",
    "robust_zscore",
    "shrinkage_whitening",
    "car",
    "bandpass",
}

FEATURES = {
    "bandpower",
    "differential_entropy",
    "hjorth",
    "connectivity",
    "riemannian_tangent",
}

AUGMENTATION = {
    "split_first_fixed_crops",
    "split_first_sliding_window",
}

TRAINERS = {
    "torch_classifier",
}

LOSSES = {
    "cross_entropy",
}

OPTIMIZERS = {
    "adam",
    "adamw",
    "sgd",
}

TORCH_BACKBONES = {
    "eegnet",
    "conformer_lite",
    "deformer_lite",
    "srfnet",
    "shallow_conv_net",
    "cbramod",
    "cbramod_pretrained",
    "fbcnet",
    "fbstcnet",
    "tsception",
}

AUGMENTATION_TRANSFORMS = {
    "channel_dropout",
    "gaussian_noise",
    "time_mask",
    "time_shift",
}

SKLEARN_MODELS = {
    "random_forest",
    "extra_trees",
    "logistic_regression",
}

GRAPH_MODELS = {
    "dgcnn",
    "lggnet",
}

SCORE_ROUTE_MODELS = {
    "score_fusion",
}

TOY_MODELS = {
    "toy_centroid",
}

MODELS = TORCH_BACKBONES | SKLEARN_MODELS | GRAPH_MODELS | SCORE_ROUTE_MODELS | TOY_MODELS

ADAPTATION = {
    "none",
    "adabn",
    "dann",
    "coral",
}

EVALUATION_PROTOCOLS = {
    "p1_repeated_group_kfold",
    "p2_pseudo_public_holdout",
    "p3_nested_selection",
}

PRIMARY_METRICS = {
    "exact_single_crop_expected_BA",
    "top4_BA",
    "no_top4_BA",
    "all_correct_rate",
}
