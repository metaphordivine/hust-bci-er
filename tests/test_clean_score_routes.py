from hust_bci_er.inference.clean_score_routes import CLEAN_SCORE_ROUTES


def test_no_trial_prior_routes_in_clean_registry():
    assert all("trial_prior_component" not in route.components for route in CLEAN_SCORE_ROUTES)
    assert all(route.candidate_allowed for route in CLEAN_SCORE_ROUTES)


def test_descriptive_score_fusion_routes_are_present():
    route_ids = {route.route_id for route in CLEAN_SCORE_ROUTES}
    assert "whitening_eps1e3_query_eps3e4_srfnet_context_fusion" in route_ids
    assert "whitening_eps1e3_with_conformer_srfnet_reference_average" in route_ids
