"""Component-to-route mapping for score-fusion orchestration.

Each score_fusion route declares logical component IDs (e.g.
``fixed_crop_ea_fbstcnet_component``) in its ``model.components`` list.
The orchestrator needs to know which base torch_classifier route produces
the predictions for each component so it can export component-score tables.

Component IDs that do not correspond to a trainable torch_classifier route
(such as handcrafted-feature components) map to ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ComponentRouteBinding:
    component_id: str
    base_route_id: str | None
    notes: str


# Known component-to-route bindings derived from clean_score_routes and
# existing route config evidence.
_COMPONENT_BINDINGS: tuple[ComponentRouteBinding, ...] = (
    ComponentRouteBinding(
        "conformer_component",
        "sliding_window_conformer_lite",
        "Conformer Lite sliding-window base route",
    ),
    ComponentRouteBinding(
        "srfnet_long_component",
        "sliding_window_srfnet",
        "SRFNet sliding-window base route",
    ),
    ComponentRouteBinding(
        "fixed_crop_ea_fbstcnet_component",
        "fixed_crop_ea_fbstcnet",
        "FBSTCNet fixed-crop EA base route",
    ),
    ComponentRouteBinding(
        "dgcnn_dann_cohort_component",
        "sliding_ea_dgcnn_dann_cohort_w6_s1",
        "Sliding-window DGCNN with cohort-domain DANN adaptation",
    ),
    ComponentRouteBinding(
        "dgcnn_coral_cohort_component",
        "sliding_ea_dgcnn_coral_cohort_w6_s1",
        "Sliding-window DGCNN with cohort-domain CORAL adaptation",
    ),
    ComponentRouteBinding(
        "riemannian_tangent_component",
        "fixed_crop_ea_riemannian_tangent",
        "Fixed-crop EA log-covariance tangent-space linear component",
    ),
    # Whitening components require SRFNet + whitening preprocessing.
    ComponentRouteBinding(
        "srfnet_whitening_eps1e3_component",
        None,
        "SRFNet with whitening_eps1e3 preprocessing; needs a dedicated route config or manually materialized run",
    ),
    ComponentRouteBinding(
        "srfnet_whitening_eps3e4_component",
        "sliding_window_srfnet_whitening_eps3e4",
        "SRFNet sliding-window route with whitening_eps3e4 preprocessing",
    ),
    # Handcrafted features are produced by feature extraction, not model training.
    ComponentRouteBinding(
        "handcrafted_feature_component",
        None,
        "Handcrafted Riemannian/bandpower features; no model training required; needs an external CSV producer",
    ),
)


def component_route_map() -> dict[str, str | None]:
    return {b.component_id: b.base_route_id for b in _COMPONENT_BINDINGS}


def component_bindings() -> Mapping[str, ComponentRouteBinding]:
    return {b.component_id: b for b in _COMPONENT_BINDINGS}


def base_route_for_component(component_id: str) -> str | None:
    for b in _COMPONENT_BINDINGS:
        if b.component_id == component_id:
            return b.base_route_id
    return None
