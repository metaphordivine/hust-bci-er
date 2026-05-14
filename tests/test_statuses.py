from pathlib import Path

import yaml

from hust_bci_er.config.registry import ROUTE_STATUSES


def test_statuses_have_single_declared_source():
    with Path("configs/statuses.yaml").open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert set(data["statuses"]) == ROUTE_STATUSES


def test_routes_are_peers_by_directory_and_initial_status():
    routes_root = Path("configs/routes")
    tier_dirs = {"stable", "candidates", "experimental", "deprecated"}
    existing = {p.name for p in routes_root.iterdir() if p.is_dir()}
    assert existing.isdisjoint(tier_dirs)
    assert (routes_root / "models").is_dir()
    for path in (routes_root / "models").glob("*.yaml"):
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["status"] == "IDEA"
        assert "tier" not in data
