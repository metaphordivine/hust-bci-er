# car_fbstcnet_as_query_noea_eps3e4_m_conn_srfnet_whitening_eps3e4_conformer_context_fusion Summary

route_id: car_fbstcnet_as_query_noea_eps3e4_m_conn_srfnet_whitening_eps3e4_conformer_context_fusion
route_status: IDEA
audit_decision: BLOCKED
gate:
primary_metric: exact_single_crop_expected_BA
decision: pending genuine component evidence
reproduce: python scripts/run_score_fusion_routes.py --route-filter car_fbstcnet_as_query_noea_eps3e4_m_conn_srfnet_whitening_eps3e4_conformer_context_fusion --export-missing --audit
dataset: train_v1
split: component protocol splits
seed: component manifests
protocol: score_fusion query_context
risk notes: Fixed alpha and temperature are design priors. Do not tune alpha or temperature from P3 outer/final results; missing or synthetic component evidence blocks the route.

## Route Family

This route is the query-context repair of the PR45 no-EA m-conn fusion screen.
The CAR FBSTCNet component remains the query, while no-EA eps3e4 m-conn
FBSTCNet, SRFNet whitening eps3e4, and Conformer act as context components.
It tests whether the no-EA m-conn signal is useful as a contextual stabilizer
rather than as a flat calibrated-average component.

## Promotion Checks

- Require genuine score matrices and source manifests for every component.
- Lightweight gate: P1 seed42 >= 0.7178, P2 mean >= 0.7037, or P2 worst > 0.5208.
- DEP/HC must be >= 0.80 to continue and should be >= 0.82 before P3 expansion.
- P3 nested selection may only use inner folds for any future alpha or temperature tuning; outer/final folds are report-only.
- Stop if the no-EA m-conn context lowers DEP/HC relative to the PR41/PR46 CAR-query context siblings.
