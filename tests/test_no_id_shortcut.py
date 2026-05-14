from pathlib import Path
from typing import Any

import yaml


ID_FEATURES = {"subject_id", "user_id", "trial_id", "pseudo_trial_id", "filename", "file_name"}
FEATURE_KEYS = {"features", "extra_features", "metadata_features", "input_features", "model_features"}


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def collect_feature_values(value: Any, *, active: bool = False) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            next_active = active or str(key) in FEATURE_KEYS
            found.extend(collect_feature_values(item, active=next_active))
    elif isinstance(value, list):
        for item in value:
            found.extend(collect_feature_values(item, active=active))
    elif active and isinstance(value, str):
        found.append(value)
    return found


def test_route_configs_do_not_use_id_features():
    hits = []
    for path in Path("configs/routes/models").glob("*.yaml"):
        data = load_yaml(path)
        feature_values = {item.lower() for item in collect_feature_values(data)}
        bad = sorted(feature_values.intersection(ID_FEATURES))
        if bad:
            hits.append(f"{path}: {bad}")
    assert hits == []
