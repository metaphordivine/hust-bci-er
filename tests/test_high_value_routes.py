from hust_bci_er.inference.high_value_routes import HIGH_VALUE_ROUTES


def test_no_c4_routes_in_clean_registry():
    assert all("c4_prior_component" not in route.components for route in HIGH_VALUE_ROUTES)
    assert all(route.candidate_allowed for route in HIGH_VALUE_ROUTES)


def test_no_c4_qkv_and_old_whitening_routes_are_present():
    route_ids = {route.route_id for route in HIGH_VALUE_ROUTES}
    assert "qkv_best_no_c4" in route_ids
    assert "old_whitening_no_c4" in route_ids
