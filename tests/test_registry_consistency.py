from pathlib import Path
import ast

import yaml

from hust_bci_er.config import registry


def builder_keys_from_source() -> set[str]:
    tree = ast.parse(Path("src/hust_bci_er/models/factory.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "BUILDERS" in names and isinstance(node.value, ast.Dict):
                return {key.value for key in node.value.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    raise AssertionError("BUILDERS mapping not found in model factory")


def test_model_registry_matches_buildable_backbones():
    assert builder_keys_from_source() == registry.TORCH_BACKBONES
    assert registry.MODELS == registry.TORCH_BACKBONES | registry.SKLEARN_MODELS | registry.GRAPH_MODELS | registry.SCORE_ROUTE_MODELS | registry.TOY_MODELS
    assert registry.SCORE_ROUTE_MODELS == {"score_fusion"}


def test_registered_protocols_have_config_files():
    for name in registry.EVALUATION_PROTOCOLS:
        assert Path("configs/protocols", f"{name}.yaml").exists()


def test_registered_statuses_and_decisions_load_from_yaml():
    statuses = yaml.safe_load(Path("configs/statuses.yaml").read_text(encoding="utf-8"))
    decisions = yaml.safe_load(Path("configs/audit_decisions.yaml").read_text(encoding="utf-8"))
    assert set(statuses["statuses"]) == registry.ROUTE_STATUSES
    assert set(decisions["decisions"]) == registry.AUDIT_DECISIONS


def test_primary_metrics_are_implemented():
    implemented = {
        "exact_single_crop_expected_BA",
        "top4_BA",
        "no_top4_BA",
        "all_correct_rate",
    }
    assert registry.PRIMARY_METRICS == implemented
