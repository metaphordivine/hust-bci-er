"""High-value clean score-route DAG declarations.

Routes that require C4/trial-prior components are intentionally omitted from
this clean repository. Keep C4 ablations outside the clean route registry.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreRoute:
    route_id: str
    score_node: str
    components: tuple[str, ...]
    candidate_allowed: bool
    notes: str


HIGH_VALUE_ROUTES = [
    ScoreRoute("s9_z_avg", "s9_score", ("conf_component", "srf_long_component"), True, "safe reference route"),
    ScoreRoute("eps3e4_conf_zavg", "eps3e4_score", ("conf_component", "eps3e4_component"), True, "whitening EPS=3e-4"),
    ScoreRoute("eps1e3_conf_zavg", "eps1e3_score", ("conf_component", "eps1e3_component"), True, "whitening EPS=1e-3"),
    ScoreRoute("h025_eps1e3_heuristic_blend", "h025_score", ("conf_component", "eps1e3_component", "heuristic_component"), True, "heuristic blend h=0.25"),
    ScoreRoute("h030_eps1e3_heuristic_blend", "h030_score", ("conf_component", "eps1e3_component", "heuristic_component"), True, "heuristic blend h=0.30"),
    ScoreRoute("h040_eps1e3_heuristic_blend", "h040_score", ("conf_component", "eps1e3_component", "heuristic_component"), True, "heuristic blend h=0.40"),
    ScoreRoute("old_whitening_no_c4", "old_whitening_no_c4_score", ("conf_component", "srf_long_component", "eps1e3_component"), True, "old+whitening with C4 removed"),
    ScoreRoute("qkv_best_no_c4", "qkv_best_no_c4_score", ("eps1e3_component", "eps3e4_component", "srf_long_component"), True, "QKV-style score fusion with C4 removed"),
]
