from hust_bci_er.inference.high_value_routes import HIGH_VALUE_ROUTES


def test_no_trial_prior_routes_in_clean_registry():
    assert all("trial_prior_component" not in route.components for route in HIGH_VALUE_ROUTES)
    assert all(route.candidate_allowed for route in HIGH_VALUE_ROUTES)


def test_clean_qkv_and_old_whitening_routes_are_present():
    route_ids = {route.route_id for route in HIGH_VALUE_ROUTES}
    assert "qkv_best_clean" in route_ids
    assert "old_whitening_clean" in route_ids
