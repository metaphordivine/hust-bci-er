"""Clean score-route DAG declarations.

Routes that require trial-prior components are intentionally omitted from this
clean repository. Keep trial-prior ablations outside the clean route registry.
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


CLEAN_SCORE_ROUTES = [
    ScoreRoute("conformer_srfnet_score_average", "conformer_srfnet_score_average", ("conformer_component", "srfnet_long_component"), True, "average of conformer and srfnet reference scores"),
    ScoreRoute("conformer_srfnet_whitening_eps3e4_average", "conformer_srfnet_whitening_eps3e4_average", ("conformer_component", "srfnet_whitening_eps3e4_component"), True, "average of conformer and srfnet whitening eps3e4 scores"),
    ScoreRoute("conformer_srfnet_whitening_eps1e3_average", "conformer_srfnet_whitening_eps1e3_average", ("conformer_component", "srfnet_whitening_eps1e3_component"), True, "average of conformer and srfnet whitening eps1e3 scores"),
    ScoreRoute("conformer_srfnet_whitening_eps1e3_handcrafted_weight25_average", "conformer_srfnet_whitening_eps1e3_handcrafted_weight25_average", ("conformer_component", "srfnet_whitening_eps1e3_component", "handcrafted_feature_component"), True, "weighted average with 25 percent handcrafted-feature score weight"),
    ScoreRoute("conformer_srfnet_whitening_eps1e3_handcrafted_weight30_average", "conformer_srfnet_whitening_eps1e3_handcrafted_weight30_average", ("conformer_component", "srfnet_whitening_eps1e3_component", "handcrafted_feature_component"), True, "weighted average with 30 percent handcrafted-feature score weight"),
    ScoreRoute("conformer_srfnet_whitening_eps1e3_handcrafted_weight40_average", "conformer_srfnet_whitening_eps1e3_handcrafted_weight40_average", ("conformer_component", "srfnet_whitening_eps1e3_component", "handcrafted_feature_component"), True, "weighted average with 40 percent handcrafted-feature score weight"),
    ScoreRoute("whitening_eps1e3_with_conformer_srfnet_reference_average", "whitening_eps1e3_with_conformer_srfnet_reference_average", ("conformer_component", "srfnet_long_component", "srfnet_whitening_eps1e3_component"), True, "average combining whitening eps1e3 scores with conformer and srfnet reference scores"),
    ScoreRoute("whitening_eps1e3_query_eps3e4_srfnet_context_fusion", "whitening_eps1e3_query_eps3e4_srfnet_context_fusion", ("srfnet_whitening_eps1e3_component", "srfnet_whitening_eps3e4_component", "srfnet_long_component"), True, "query-context fusion using whitening eps1e3 as query and eps3e4/srfnet reference as context"),
]
