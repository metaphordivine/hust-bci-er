"""Clean score-route DAG declarations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreRoute:
    route_id: str
    score_node: str
    components: tuple[str, ...]
    review_allowed: bool
    notes: str
    method: str = "z_average"
    weights: tuple[float, ...] | None = None
    query_component: str | None = None
    context_components: tuple[str, ...] = ()
    alpha: float | None = None
    temperature: float | None = None
    adaptive_alpha_max: float | None = None
    margin_low: float | None = None
    margin_high: float | None = None


CLEAN_SCORE_ROUTES = [
    ScoreRoute("conformer_srfnet_score_average", "conformer_srfnet_score_average", ("conformer_component", "srfnet_long_component"), True, "average of conformer and srfnet reference scores"),
    ScoreRoute("conformer_srfnet_whitening_eps3e4_average", "conformer_srfnet_whitening_eps3e4_average", ("conformer_component", "srfnet_whitening_eps3e4_component"), True, "average of conformer and srfnet whitening eps3e4 scores"),
    ScoreRoute("conformer_srfnet_whitening_eps1e3_average", "conformer_srfnet_whitening_eps1e3_average", ("conformer_component", "srfnet_whitening_eps1e3_component"), True, "average of conformer and srfnet whitening eps1e3 scores"),
    ScoreRoute("conformer_srfnet_whitening_eps1e3_handcrafted_weight25_average", "conformer_srfnet_whitening_eps1e3_handcrafted_weight25_average", ("conformer_component", "srfnet_whitening_eps1e3_component", "handcrafted_feature_component"), True, "weighted average with 25 percent handcrafted-feature score weight", weights=(0.375, 0.375, 0.25)),
    ScoreRoute("conformer_srfnet_whitening_eps1e3_handcrafted_weight30_average", "conformer_srfnet_whitening_eps1e3_handcrafted_weight30_average", ("conformer_component", "srfnet_whitening_eps1e3_component", "handcrafted_feature_component"), True, "weighted average with 30 percent handcrafted-feature score weight", weights=(0.35, 0.35, 0.30)),
    ScoreRoute("conformer_srfnet_whitening_eps1e3_handcrafted_weight40_average", "conformer_srfnet_whitening_eps1e3_handcrafted_weight40_average", ("conformer_component", "srfnet_whitening_eps1e3_component", "handcrafted_feature_component"), True, "weighted average with 40 percent handcrafted-feature score weight", weights=(0.30, 0.30, 0.40)),
    ScoreRoute("whitening_eps1e3_with_conformer_srfnet_reference_average", "whitening_eps1e3_with_conformer_srfnet_reference_average", ("conformer_component", "srfnet_long_component", "srfnet_whitening_eps1e3_component"), True, "average combining whitening eps1e3 scores with conformer and srfnet reference scores"),
    ScoreRoute(
        "fbstcnet_srfnet_score_average",
        "fbstcnet_srfnet_score_average",
        ("fixed_crop_ea_fbstcnet_component", "srfnet_long_component"),
        True,
        "weighted z-average of fixed-crop EA FBSTCNet and sliding-window SRFNet scores",
        weights=(0.50, 0.50),
    ),
    ScoreRoute(
        "fbstcnet_srfnet_conformer_score_average",
        "fbstcnet_srfnet_conformer_score_average",
        ("fixed_crop_ea_fbstcnet_component", "srfnet_long_component", "conformer_component"),
        True,
        "weighted z-average of fixed-crop EA FBSTCNet, sliding-window SRFNet, and Conformer scores",
        weights=(0.40, 0.40, 0.20),
    ),
    ScoreRoute(
        "dgcnn_adaptation_fbstcnet_srfnet_conformer_score_average",
        "dgcnn_adaptation_fbstcnet_srfnet_conformer_score_average",
        (
            "dgcnn_dann_cohort_component",
            "dgcnn_coral_cohort_component",
            "fixed_crop_ea_fbstcnet_component",
            "srfnet_long_component",
            "conformer_component",
        ),
        True,
        "weighted z-average of cohort-adapted DGCNN components with FBSTCNet, SRFNet, and Conformer scores",
        weights=(0.20, 0.20, 0.25, 0.20, 0.15),
    ),
    ScoreRoute(
        "interpretable_calibrated_diverse_score_fusion",
        "interpretable_calibrated_diverse_score_fusion",
        (
            "fixed_crop_ea_fbstcnet_component",
            "srfnet_long_component",
            "conformer_component",
            "dgcnn_dann_cohort_component",
            "dgcnn_coral_cohort_component",
            "riemannian_tangent_component",
        ),
        True,
        "temperature-smoothed probability average across FBSTCNet, SRFNet, Conformer, DGCNN adaptation, and Riemannian tangent components",
        method="calibrated_probability_average",
        weights=(0.24, 0.22, 0.18, 0.14, 0.14, 0.08),
        temperature=1.25,
    ),
    ScoreRoute(
        "fbstcnet_srfnet_whitening_eps1e3_average",
        "fbstcnet_srfnet_whitening_eps1e3_average",
        ("fixed_crop_ea_fbstcnet_component", "srfnet_long_component", "srfnet_whitening_eps1e3_component"),
        True,
        "weighted z-average of fixed-crop EA FBSTCNet, SRFNet reference, and whitening eps1e3 SRFNet scores",
        weights=(0.40, 0.30, 0.30),
    ),
    ScoreRoute(
        "fbstcnet_as_query_srfnet_context_fusion",
        "fbstcnet_as_query_srfnet_context_fusion",
        ("fixed_crop_ea_fbstcnet_component", "srfnet_long_component", "conformer_component"),
        True,
        "query-context fusion using fixed-crop EA FBSTCNet as query and SRFNet/Conformer as context",
        method="query_context",
        query_component="fixed_crop_ea_fbstcnet_component",
        context_components=("srfnet_long_component", "conformer_component"),
        alpha=0.25,
        temperature=0.75,
    ),
    ScoreRoute(
        "car_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion",
        "car_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion",
        ("fixed_crop_car_fbstcnet_component", "srfnet_whitening_eps3e4_component", "conformer_component"),
        True,
        "query-context fusion using fixed-crop CAR FBSTCNet as query and SRFNet whitening eps3e4/Conformer as context",
        method="query_context",
        query_component="fixed_crop_car_fbstcnet_component",
        context_components=("srfnet_whitening_eps3e4_component", "conformer_component"),
        alpha=0.25,
        temperature=0.75,
    ),
    ScoreRoute(
        "car_fbstcnet_as_query_srfnet_reference_whitening_eps3e4_conformer_context_fusion",
        "car_fbstcnet_as_query_srfnet_reference_whitening_eps3e4_conformer_context_fusion",
        (
            "fixed_crop_car_fbstcnet_component",
            "srfnet_long_component",
            "srfnet_whitening_eps3e4_component",
            "conformer_component",
        ),
        True,
        "query-context fusion using fixed-crop CAR FBSTCNet as query and SRFNet reference, SRFNet whitening eps3e4, and Conformer as context",
        method="query_context",
        query_component="fixed_crop_car_fbstcnet_component",
        context_components=("srfnet_long_component", "srfnet_whitening_eps3e4_component", "conformer_component"),
        alpha=0.25,
        temperature=0.75,
    ),
    ScoreRoute(
        "car_fbstcnet_margin_adaptive_srfnet_whitening_eps3e4_conformer_context_fusion",
        "car_fbstcnet_margin_adaptive_srfnet_whitening_eps3e4_conformer_context_fusion",
        ("fixed_crop_car_fbstcnet_component", "srfnet_whitening_eps3e4_component", "conformer_component"),
        True,
        "margin-adaptive query-context fusion that increases SRFNet/Conformer context weight when CAR FBSTCNet top-4 margin is small",
        method="margin_adaptive_query_context",
        query_component="fixed_crop_car_fbstcnet_component",
        context_components=("srfnet_whitening_eps3e4_component", "conformer_component"),
        alpha=0.25,
        adaptive_alpha_max=0.45,
        temperature=0.75,
        margin_low=0.15,
        margin_high=0.85,
    ),
    ScoreRoute(
        "car_fbstcnet_query_srfnet_whitening_eps3e4_conformer_tuned_fbstcnet_context_fusion",
        "car_fbstcnet_query_srfnet_whitening_eps3e4_conformer_tuned_fbstcnet_context_fusion",
        (
            "fixed_crop_car_fbstcnet_component",
            "srfnet_whitening_eps3e4_component",
            "conformer_component",
            "tuned_sliding_fbstcnet_zscore_component",
        ),
        True,
        "query-context fusion using fixed-crop CAR FBSTCNet as query and SRFNet whitening eps3e4/Conformer/tuned sliding FBSTCNet zscore as context",
        method="query_context",
        query_component="fixed_crop_car_fbstcnet_component",
        context_components=(
            "srfnet_whitening_eps3e4_component",
            "conformer_component",
            "tuned_sliding_fbstcnet_zscore_component",
        ),
        alpha=0.25,
        temperature=0.75,
    ),
    ScoreRoute(
        "whitening_eps3e4_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion",
        "whitening_eps3e4_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion",
        ("fixed_crop_whitening_eps3e4_fbstcnet_component", "srfnet_whitening_eps3e4_component", "conformer_component"),
        True,
        "query-context fusion using no-EA eps3e4 FBSTCNet as query and SRFNet whitening eps3e4/Conformer as context",
        method="query_context",
        query_component="fixed_crop_whitening_eps3e4_fbstcnet_component",
        context_components=("srfnet_whitening_eps3e4_component", "conformer_component"),
        alpha=0.25,
        temperature=0.75,
    ),
    ScoreRoute(
        "car_whitening_eps3e4_fbstcnet_srfnet_whitening_eps3e4_conformer_score_average",
        "car_whitening_eps3e4_fbstcnet_srfnet_whitening_eps3e4_conformer_score_average",
        (
            "fixed_crop_car_fbstcnet_component",
            "fixed_crop_whitening_eps3e4_fbstcnet_component",
            "srfnet_whitening_eps3e4_component",
            "conformer_component",
        ),
        True,
        "weighted z-average of CAR FBSTCNet, no-EA eps3e4 FBSTCNet, SRFNet whitening eps3e4, and Conformer scores",
        weights=(0.35, 0.25, 0.25, 0.15),
    ),
    ScoreRoute(
        "car_noea_eps3e4_m_conn_srfnet_whitening_conformer_calibrated_average",
        "car_noea_eps3e4_m_conn_srfnet_whitening_conformer_calibrated_average",
        (
            "fixed_crop_car_fbstcnet_component",
            "fixed_crop_whitening_eps3e4_fbstcnet_m_conn_component",
            "srfnet_whitening_eps3e4_component",
            "conformer_component",
        ),
        True,
        "temperature-smoothed probability average of CAR FBSTCNet, no-EA eps3e4 m-conn FBSTCNet, SRFNet whitening eps3e4, and Conformer scores",
        method="calibrated_probability_average",
        weights=(0.34, 0.26, 0.24, 0.16),
        temperature=1.20,
    ),
    ScoreRoute(
        "car_fbstcnet_as_query_noea_eps3e4_m_conn_srfnet_whitening_eps3e4_conformer_context_fusion",
        "car_fbstcnet_as_query_noea_eps3e4_m_conn_srfnet_whitening_eps3e4_conformer_context_fusion",
        (
            "fixed_crop_car_fbstcnet_component",
            "fixed_crop_whitening_eps3e4_fbstcnet_m_conn_component",
            "srfnet_whitening_eps3e4_component",
            "conformer_component",
        ),
        True,
        "query-context fusion using fixed-crop CAR FBSTCNet as query and no-EA eps3e4 m-conn FBSTCNet, SRFNet whitening eps3e4, and Conformer as context",
        method="query_context",
        query_component="fixed_crop_car_fbstcnet_component",
        context_components=(
            "fixed_crop_whitening_eps3e4_fbstcnet_m_conn_component",
            "srfnet_whitening_eps3e4_component",
            "conformer_component",
        ),
        alpha=0.25,
        temperature=0.75,
    ),
    ScoreRoute(
        "srfnet_as_query_fbstcnet_context_fusion",
        "srfnet_as_query_fbstcnet_context_fusion",
        ("srfnet_long_component", "fixed_crop_ea_fbstcnet_component", "conformer_component"),
        True,
        "query-context fusion using SRFNet as query and fixed-crop EA FBSTCNet/Conformer as context",
        method="query_context",
        query_component="srfnet_long_component",
        context_components=("fixed_crop_ea_fbstcnet_component", "conformer_component"),
        alpha=0.25,
        temperature=0.75,
    ),
    ScoreRoute(
        "whitening_eps1e3_query_eps3e4_srfnet_context_fusion",
        "whitening_eps1e3_query_eps3e4_srfnet_context_fusion",
        ("srfnet_whitening_eps1e3_component", "srfnet_whitening_eps3e4_component", "srfnet_long_component"),
        True,
        "query-context fusion using whitening eps1e3 as query and eps3e4/srfnet reference as context",
        method="query_context",
        query_component="srfnet_whitening_eps1e3_component",
        context_components=("srfnet_whitening_eps3e4_component", "srfnet_long_component"),
        alpha=0.25,
        temperature=0.75,
    ),
]


def score_route_by_id(route_id: str) -> ScoreRoute | None:
    for route in CLEAN_SCORE_ROUTES:
        if route.route_id == route_id:
            return route
    return None


def score_route_by_node(score_node: str) -> ScoreRoute | None:
    for route in CLEAN_SCORE_ROUTES:
        if route.score_node == score_node:
            return route
    return None
