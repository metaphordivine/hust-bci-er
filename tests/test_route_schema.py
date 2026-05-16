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


def test_sliding_window_fbstcnet_route_declares_cheby2_filterbank():
    """Route config must explicitly declare filterbank_type: cheby2 (paper-faithful)."""
    data = load_route("configs/routes/models/sliding_window_fbstcnet.yaml")
    model_cfg = data.get("model", {})
    assert model_cfg.get("filterbank_type") == "cheby2", (
        "sliding_window_fbstcnet route must explicitly declare filterbank_type: cheby2"
    )


def test_sliding_window_augmentation_requires_split_first_and_matching_window():
    data = load_route("configs/routes/models/sliding_window_eegnet.yaml")
    assert validate_route_config(data, path=Path("sliding_window_eegnet.yaml")) == []

    data["augmentation"] = dict(data["augmentation"])
    data["augmentation"]["split_first"] = False
    errors = validate_route_config(data, path=Path("sliding_window_eegnet.yaml"))
    assert any("split_first" in err for err in errors)

    data["augmentation"]["split_first"] = True
    data["input_window_sec"] = 10
    errors = validate_route_config(data, path=Path("sliding_window_eegnet.yaml"))
    assert any("input_window_sec must match" in err for err in errors)


def test_training_config_rejects_unknown_optimizer():
    data = load_route("configs/routes/models/sliding_window_eegnet.yaml")
    data["training"] = dict(data["training"])
    data["training"]["optimizer"] = dict(data["training"]["optimizer"])
    data["training"]["optimizer"]["name"] = "rmsprop"
    errors = validate_route_config(data, path=Path("sliding_window_eegnet.yaml"))
    assert any("unknown optimizer" in err for err in errors)


def test_sliding_window_search_space_is_validated():
    data = load_route("configs/routes/models/sliding_window_eegnet.yaml")
    data["augmentation"] = dict(data["augmentation"])
    data["augmentation"]["search_space"] = {"window_sec": [4, 12]}
    errors = validate_route_config(data, path=Path("sliding_window_eegnet.yaml"))
    assert any("search_space.window_sec" in err for err in errors)


def test_sliding_window_search_space_cross_product_must_match_exact_metric_shape():
    data = load_route("configs/routes/models/sliding_window_eegnet.yaml")
    data["augmentation"] = dict(data["augmentation"])
    data["augmentation"]["window_sec"] = 5
    data["input_window_sec"] = 5
    data["augmentation"]["search_space"] = {
        "source_trial_sec": [10],
        "window_sec": [5],
        "stride_sec": [1, 2],
    }

    errors = validate_route_config(data, path=Path("sliding_window_eegnet.yaml"))

    assert any("METRIC_SCORE_MATRIX_SHAPE_COMPATIBLE" in err for err in errors)


def test_sliding_window_search_space_rejects_input_window_mismatch():
    data = load_route("configs/routes/models/sliding_window_eegnet.yaml")
    data["augmentation"] = dict(data["augmentation"])
    data["augmentation"]["search_space"] = {"window_sec": [4]}

    errors = validate_route_config(data, path=Path("sliding_window_eegnet.yaml"))

    assert any("SEARCH_SPACE_CROSS_PRODUCT_VALID" in err for err in errors)


def test_augmentation_search_space_rejects_fields_for_other_augmentation_type():
    data = load_route("configs/routes/models/sliding_window_eegnet.yaml")
    data["augmentation"] = dict(data["augmentation"])
    data["augmentation"]["search_space"] = {"n_crops": [1]}
    errors = validate_route_config(data, path=Path("sliding_window_eegnet.yaml"))
    assert any("unsupported field: n_crops" in err for err in errors)

    data = load_route("configs/routes/models/fixed_crop_pure_deformer_lite.yaml")
    data["augmentation"] = dict(data["augmentation"])
    data["augmentation"]["search_space"] = {"stride_sec": [1]}
    errors = validate_route_config(data, path=Path("fixed_crop_pure_deformer_lite.yaml"))
    assert any("unsupported field: stride_sec" in err for err in errors)


def test_fixed_crop_augmentation_requires_valid_crop_grid():
    data = load_route("configs/routes/models/ea_deformer.yaml")
    data["route_id"] = "fixed_crop_ea_deformer"
    data["input_window_sec"] = 10
    data["augmentation"] = {
        "name": "split_first_fixed_crops",
        "split_first": True,
        "source_trial_sec": 50,
        "window_sec": 10,
        "n_crops": 5,
        "apply_to_splits": ["train", "val", "test"],
        "aggregate_to_trial": {"method": "mean_score", "tie_break": "mean_score"},
    }
    data["inference"] = {"top4": True, "crop_policy": "exact_single_crop"}
    assert validate_route_config(data, path=Path("fixed_crop_ea_deformer.yaml")) == []

    data["augmentation"] = dict(data["augmentation"])
    data["augmentation"]["n_crops"] = 6
    errors = validate_route_config(data, path=Path("fixed_crop_ea_deformer.yaml"))
    assert any("n_crops == 5" in err for err in errors)


def test_exact_fixed_crop_route_rejects_search_space_crop_count_other_than_five():
    data = load_route("configs/routes/models/fixed_crop_pure_deformer_lite.yaml")
    data["augmentation"] = dict(data["augmentation"])
    data["augmentation"]["search_space"] = {"n_crops": [5, 6]}
    errors = validate_route_config(data, path=Path("fixed_crop_pure_deformer_lite.yaml"))
    assert any("search_space.n_crops values to be 5" in err for err in errors)


def test_fixed_crop_search_space_rejects_invalid_cross_product():
    data = load_route("configs/routes/models/fixed_crop_pure_deformer_lite.yaml")
    data["augmentation"] = dict(data["augmentation"])
    data["augmentation"]["search_space"] = {"window_sec": [30], "n_crops": [2]}
    errors = validate_route_config(data, path=Path("fixed_crop_pure_deformer_lite.yaml"))
    assert any("search_space fixed-crop combinations" in err for err in errors)


def test_fixed_crop_search_space_cross_product_uses_candidate_source_duration():
    data = load_route("configs/routes/models/fixed_crop_pure_deformer_lite.yaml")
    data["augmentation"] = dict(data["augmentation"])
    data["augmentation"]["source_trial_sec"] = 60
    data["augmentation"]["search_space"] = {
        "source_trial_sec": [50],
        "window_sec": [30],
        "n_crops": [2],
    }
    errors = validate_route_config(data, path=Path("fixed_crop_pure_deformer_lite.yaml"))
    assert any("fixed-crop combinations" in err for err in errors)
