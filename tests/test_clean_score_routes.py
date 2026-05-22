import yaml

from pathlib import Path

from hust_bci_er.config.component_map import base_route_for_component
from hust_bci_er.inference.clean_score_routes import CLEAN_SCORE_ROUTES, score_route_by_id, score_route_by_node


def test_no_trial_prior_routes_in_clean_registry():
    assert all("trial_prior_component" not in route.components for route in CLEAN_SCORE_ROUTES)
    assert all(route.review_allowed for route in CLEAN_SCORE_ROUTES)


def test_descriptive_score_fusion_routes_are_present():
    route_ids = {route.route_id for route in CLEAN_SCORE_ROUTES}
    assert "whitening_eps1e3_query_eps3e4_srfnet_context_fusion" in route_ids
    assert "whitening_eps1e3_with_conformer_srfnet_reference_average" in route_ids
    assert "fbstcnet_srfnet_score_average" in route_ids
    assert "fbstcnet_as_query_srfnet_context_fusion" in route_ids
    assert "car_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion" in route_ids
    assert "car_fbstcnet_margin_adaptive_srfnet_whitening_eps3e4_conformer_context_fusion" in route_ids
    assert "dgcnn_adaptation_fbstcnet_srfnet_conformer_score_average" in route_ids
    assert "interpretable_calibrated_diverse_score_fusion" in route_ids


def test_fbstcnet_fusion_routes_record_initial_weights():
    pair = score_route_by_id("fbstcnet_srfnet_score_average")
    assert pair is not None
    assert pair.weights == (0.50, 0.50)

    triple = score_route_by_id("fbstcnet_srfnet_conformer_score_average")
    assert triple is not None
    assert triple.weights == (0.40, 0.40, 0.20)

    whitening = score_route_by_id("fbstcnet_srfnet_whitening_eps1e3_average")
    assert whitening is not None
    assert whitening.weights == (0.40, 0.30, 0.30)


def test_dgcnn_adaptation_fusion_route_records_components_and_weights():
    route = score_route_by_id("dgcnn_adaptation_fbstcnet_srfnet_conformer_score_average")
    assert route is not None
    assert route.components == (
        "dgcnn_dann_cohort_component",
        "dgcnn_coral_cohort_component",
        "fixed_crop_ea_fbstcnet_component",
        "srfnet_long_component",
        "conformer_component",
    )
    assert route.weights == (0.20, 0.20, 0.25, 0.20, 0.15)
    assert base_route_for_component("dgcnn_dann_cohort_component") == "sliding_ea_dgcnn_dann_cohort_w6_s1"
    assert base_route_for_component("dgcnn_coral_cohort_component") == "sliding_ea_dgcnn_coral_cohort_w6_s1"


def test_interpretable_calibrated_diverse_route_records_components_and_weights():
    route = score_route_by_id("interpretable_calibrated_diverse_score_fusion")
    assert route is not None
    assert route.method == "calibrated_probability_average"
    assert route.components == (
        "fixed_crop_ea_fbstcnet_component",
        "srfnet_long_component",
        "conformer_component",
        "dgcnn_dann_cohort_component",
        "dgcnn_coral_cohort_component",
        "riemannian_tangent_component",
    )
    assert route.weights == (0.24, 0.22, 0.18, 0.14, 0.14, 0.08)
    assert route.temperature == 1.25
    assert base_route_for_component("riemannian_tangent_component") == "fixed_crop_ea_riemannian_tangent"


def test_fbstcnet_query_context_routes_are_registered():
    fbst = score_route_by_id("fbstcnet_as_query_srfnet_context_fusion")
    assert fbst is not None
    assert fbst.method == "query_context"
    assert fbst.query_component == "fixed_crop_ea_fbstcnet_component"
    assert fbst.context_components == ("srfnet_long_component", "conformer_component")
    assert fbst.alpha == 0.25
    assert fbst.temperature == 0.75

    srf = score_route_by_id("srfnet_as_query_fbstcnet_context_fusion")
    assert srf is not None
    assert srf.method == "query_context"
    assert srf.query_component == "srfnet_long_component"
    assert srf.context_components == ("fixed_crop_ea_fbstcnet_component", "conformer_component")
    assert srf.alpha == 0.25
    assert srf.temperature == 0.75


def test_car_fbstcnet_query_context_route_uses_whitening_context():
    route = score_route_by_id("car_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion")
    assert route is not None
    assert route.method == "query_context"
    assert route.components == (
        "fixed_crop_car_fbstcnet_component",
        "srfnet_whitening_eps3e4_component",
        "conformer_component",
    )
    assert route.query_component == "fixed_crop_car_fbstcnet_component"
    assert route.context_components == ("srfnet_whitening_eps3e4_component", "conformer_component")
    assert route.alpha == 0.25
    assert route.temperature == 0.75
    assert base_route_for_component("fixed_crop_car_fbstcnet_component") == "fixed_crop_car_fbstcnet"
    assert base_route_for_component("srfnet_whitening_eps3e4_component") == "sliding_window_srfnet_whitening_eps3e4"


def test_car_fbstcnet_margin_adaptive_route_uses_whitening_context():
    route = score_route_by_id("car_fbstcnet_margin_adaptive_srfnet_whitening_eps3e4_conformer_context_fusion")
    assert route is not None
    assert route.method == "margin_adaptive_query_context"
    assert route.components == (
        "fixed_crop_car_fbstcnet_component",
        "srfnet_whitening_eps3e4_component",
        "conformer_component",
    )
    assert route.query_component == "fixed_crop_car_fbstcnet_component"
    assert route.context_components == ("srfnet_whitening_eps3e4_component", "conformer_component")
    assert route.alpha == 0.25
    assert route.adaptive_alpha_max == 0.45
    assert route.temperature == 0.75
    assert route.margin_low == 0.15
    assert route.margin_high == 0.85
    assert base_route_for_component("fixed_crop_car_fbstcnet_component") == "fixed_crop_car_fbstcnet"
    assert base_route_for_component("srfnet_whitening_eps3e4_component") == "sliding_window_srfnet_whitening_eps3e4"


def test_car_noea_m_conn_calibrated_route_uses_m_conn_component():
    route = score_route_by_id("car_noea_eps3e4_m_conn_srfnet_whitening_conformer_calibrated_average")
    assert route is not None
    assert route.method == "calibrated_probability_average"
    assert route.components == (
        "fixed_crop_car_fbstcnet_component",
        "fixed_crop_whitening_eps3e4_fbstcnet_m_conn_component",
        "srfnet_whitening_eps3e4_component",
        "conformer_component",
    )
    assert route.weights == (0.34, 0.26, 0.24, 0.16)
    assert route.temperature == 1.20
    assert base_route_for_component("fixed_crop_whitening_eps3e4_fbstcnet_m_conn_component") == (
        "fixed_crop_whitening_eps3e4_fbstcnet_m_conn_light"
    )


def test_score_route_lookup_by_id_and_node():
    route = score_route_by_id("conformer_srfnet_score_average")
    assert route is not None
    assert score_route_by_node(route.score_node) == route


def test_whitening_eps3e4_component_has_current_style_base_route():
    route = score_route_by_id("conformer_srfnet_whitening_eps3e4_average")
    assert route is not None
    assert route.components == ("conformer_component", "srfnet_whitening_eps3e4_component")
    assert base_route_for_component("srfnet_whitening_eps3e4_component") == "sliding_window_srfnet_whitening_eps3e4"


def test_score_route_configs_match_clean_registry():
    for path in Path("configs/routes/models").glob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        model = data.get("model") or {}
        if model.get("name") != "score_fusion":
            continue
        route = score_route_by_id(data["route_id"])
        assert route is not None
        assert model["score_node"] == route.score_node
        assert tuple(model["components"]) == route.components
