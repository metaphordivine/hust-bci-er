import yaml

from pathlib import Path

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


def test_score_route_lookup_by_id_and_node():
    route = score_route_by_id("conformer_srfnet_score_average")
    assert route is not None
    assert score_route_by_node(route.score_node) == route


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
