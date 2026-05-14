from pathlib import Path

import yaml

from hust_bci_er.config.registry import AUDIT_DECISIONS, ROUTE_STATUSES


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def test_statuses_and_audit_decisions_have_declared_sources():
    status_data = load_yaml(Path("configs/statuses.yaml"))
    decision_data = load_yaml(Path("configs/audit_decisions.yaml"))
    assert set(status_data["statuses"]) == ROUTE_STATUSES
    assert set(decision_data["decisions"]) == AUDIT_DECISIONS
    assert "BLOCKED" not in ROUTE_STATUSES
    assert "BLOCKED" in AUDIT_DECISIONS


def test_routes_are_peers_by_directory_and_have_valid_statuses():
    routes_root = Path("configs/routes")
    tier_dirs = {"stable", "candidates", "experimental", "deprecated"}
    existing = {p.name for p in routes_root.iterdir() if p.is_dir()}
    assert existing.isdisjoint(tier_dirs)
    assert (routes_root / "models").is_dir()
    for path in (routes_root / "models").glob("*.yaml"):
        data = load_yaml(path)
        assert data["status"] in ROUTE_STATUSES
        assert "tier" not in data


def test_advanced_route_statuses_require_summary_evidence():
    statuses_requiring_summary = {"CANDIDATE", "PROMOTED", "REJECTED", "ARCHIVED"}
    for path in Path("configs/routes/models").glob("*.yaml"):
        data = load_yaml(path)
        if data.get("status") in statuses_requiring_summary:
            route_id = data["route_id"]
            assert Path("reports/route_summaries", f"{route_id}_summary.md").exists()
