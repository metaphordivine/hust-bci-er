# car_noea_eps3e4_m_conn_srfnet_whitening_conformer_calibrated_average Summary

route_id: car_noea_eps3e4_m_conn_srfnet_whitening_conformer_calibrated_average
route_status: IDEA
audit_decision: BLOCKED
gate:
primary_metric: exact_single_crop_expected_BA
decision: pending genuine component evidence
reproduce: python scripts/run_score_fusion_routes.py --route-filter car_noea_eps3e4_m_conn_srfnet_whitening_conformer_calibrated_average --export-missing --audit
dataset: train_v1
split: component protocol splits
seed: component manifests
protocol: score_fusion calibrated_probability_average
risk notes: Fixed weights and temperature are design priors. Do not tune weights from P3 outer/final results; missing or synthetic component evidence blocks the route.

## Route Family

This route extends the PR41 CAR/SRFNet/Conformer fusion family with the PR44
no-EA eps3e4 m-conn FBSTCNet control. It uses a temperature-smoothed
probability average rather than a query-only route so the new component can act
as a calibrated complement instead of replacing the CAR query.

## Promotion Checks

- Require genuine score matrices and source manifests for every component.
- Lightweight gate: P1 seed42 >= 0.7178, P2 mean >= 0.7037, or P2 worst > 0.5208.
- DEP/HC must be >= 0.80 to continue and should be >= 0.82 before P3 expansion.
- P3 nested selection may only use inner folds for any future weight or temperature tuning; outer/final folds are report-only.
- Stop if no-EA m-conn evidence is missing, synthetic, or misaligned with CAR/SRFNet/Conformer component keys.
