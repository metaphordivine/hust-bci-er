"""Named preprocessing variants."""

from __future__ import annotations

from copy import deepcopy


BASE_PREPROCESSING = {
    "bandpass": {"enabled": False, "low_hz": 1.0, "high_hz": 45.0, "order": 4},
    "notch": {"enabled": False, "freq_hz": 50.0, "quality_factor": 30.0},
    "zscore": {"enabled": True},
    "reference": {"mode": "none"},
    "channel_clip": {"enabled": False, "threshold_mad": 5.0},
    "robust_zscore": {"enabled": False, "mode": "channel"},
    "whitening": {"enabled": False, "eps": 1e-4, "shrinkage_alpha": 0.0},
}


def preprocessing_variants() -> dict[str, dict]:
    variants: dict[str, dict] = {}
    variants["original_zscore"] = deepcopy(BASE_PREPROCESSING)

    cfg = deepcopy(BASE_PREPROCESSING)
    cfg["reference"] = {"mode": "car"}
    variants["car_zscore"] = cfg

    for name, eps in [("whitening_eps1e3", 1e-3), ("whitening_eps3e4", 3e-4)]:
        cfg = deepcopy(BASE_PREPROCESSING)
        cfg["zscore"] = {"enabled": False}
        cfg["whitening"] = {"enabled": True, "eps": eps, "shrinkage_alpha": 0.0}
        cfg["robust_zscore"] = {"enabled": True, "mode": "channel"}
        variants[name] = cfg

    for name, eps, alpha in [
        ("shrinkage_whitening_eps1e3_alpha_0p10", 1e-3, 0.10),
        ("shrinkage_whitening_eps3e4_alpha_0p10", 3e-4, 0.10),
    ]:
        cfg = deepcopy(BASE_PREPROCESSING)
        cfg["zscore"] = {"enabled": False}
        cfg["whitening"] = {"enabled": True, "eps": eps, "shrinkage_alpha": alpha}
        cfg["robust_zscore"] = {"enabled": True, "mode": "channel"}
        variants[name] = cfg

    cfg = deepcopy(BASE_PREPROCESSING)
    cfg["euclidean_alignment"] = {"enabled": True, "fit_scope": "train_split_only"}
    variants["ea_zscore"] = cfg
    return variants


def get_preprocessing_variant(name: str) -> dict:
    variants = preprocessing_variants()
    if name not in variants:
        raise KeyError(f"unknown preprocessing variant: {name}")
    return deepcopy(variants[name])
