# car_whitening_eps3e4_fbstcnet_srfnet_whitening_eps3e4_conformer_score_average Summary

route_id: car_whitening_eps3e4_fbstcnet_srfnet_whitening_eps3e4_conformer_score_average
route_status: IDEA
audit_decision: BLOCKED
gate:
primary_metric: exact_single_crop_expected_BA
decision: pending genuine component evidence
reproduce: python scripts/run_score_fusion_routes.py --route-filter car_whitening_eps3e4_fbstcnet_srfnet_whitening_eps3e4_conformer_score_average --export-missing --audit
dataset: train_v1
split: component protocol splits
seed: component manifests
protocol: score_fusion weighted z-average
risk notes: Fixed weights are design priors. Do not reinterpret this as calibrated evidence until component-level P1/P2/P3 runs are auditable.

## Route Family

This route combines CAR FBSTCNet, no-EA eps3e4 FBSTCNet, SRFNet whitening
eps3e4, and Conformer score matrices with fixed weights. It is designed as a
diverse score-fusion screen rather than a new backbone.

## Promotion Checks

- Require genuine score matrices and source manifests for every component.
- Compare against the existing CAR query-context score route.
- Promote only if P1/P2 evidence improves without worsening DEP/HC balance.
