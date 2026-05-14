from pathlib import Path

import yaml

from hust_bci_er.config.schema import validate_route_config
from scripts.validate_route import validate_file


def test_example_route_schema_passes():
    path = Path("configs/routes/models/ea_deformer.yaml")
    assert validate_file(path) == []


def load_route(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_score_fusion_route_requires_registered_score_node_and_components():
    data = load_route("configs/routes/models/conformer_srfnet_score_average.yaml")
    assert validate_route_config(data, path=Path("conformer_srfnet_score_average.yaml")) == []

    missing_components = dict(data)
    missing_components["model"] = {"name": "score_fusion", "score_node": "conformer_srfnet_score_average"}
    errors = validate_route_config(missing_components, path=Path("conformer_srfnet_score_average.yaml"))
    assert any("model.components" in err for err in errors)

    wrong_node = dict(data)
    wrong_node["model"] = dict(data["model"])
    wrong_node["model"]["score_node"] = "wrong_node"
    errors = validate_route_config(wrong_node, path=Path("conformer_srfnet_score_average.yaml"))
    assert any("score_node must match" in err for err in errors)


def test_non_score_fusion_routes_cannot_declare_score_nodes():
    data = load_route("configs/routes/models/ea_deformer.yaml")
    data["model"] = {"name": "deformer_lite", "score_node": "not_allowed"}
    errors = validate_route_config(data, path=Path("ea_deformer.yaml"))
    assert any("only allowed for score_fusion" in err for err in errors)


def test_primary_metric_must_match_top4_policy():
    data = load_route("configs/routes/models/ea_deformer.yaml")
    data["evaluation"] = dict(data["evaluation"])
    data["evaluation"]["primary_metric"] = "no_top4_BA"
    errors = validate_route_config(data, path=Path("ea_deformer.yaml"))
    assert any("no_top4_BA requires inference.top4: false" in err for err in errors)

    data["evaluation"]["primary_metric"] = "top4_BA"
    data["inference"] = dict(data["inference"])
    data["inference"]["top4"] = False
    errors = validate_route_config(data, path=Path("ea_deformer.yaml"))
    assert any("top4_BA requires inference.top4: true" in err for err in errors)


def test_route_config_must_not_set_output_dir():
    data = load_route("configs/routes/models/ea_deformer.yaml")
    data["output_dir"] = "outputs/manual"
    errors = validate_route_config(data, path=Path("ea_deformer.yaml"))
    assert any("output_dir is derived" in err for err in errors)
