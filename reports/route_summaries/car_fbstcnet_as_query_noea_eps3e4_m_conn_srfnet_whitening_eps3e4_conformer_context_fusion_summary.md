# car_fbstcnet_as_query_noea_eps3e4_m_conn_srfnet_whitening_eps3e4_conformer_context_fusion Summary

route_id: car_fbstcnet_as_query_noea_eps3e4_m_conn_srfnet_whitening_eps3e4_conformer_context_fusion
route_status: IDEA
audit_decision: PASS_P1_P2_FIXED_CROPS_AND_P3
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.7365558186666668
decision: KEEP_AS_PR41_SIBLING_NOT_NEW_CHAMPION
reproduce: python scripts/run_score_fusion_routes.py --route-filter car_fbstcnet_as_query_noea_eps3e4_m_conn_srfnet_whitening_eps3e4_conformer_context_fusion --export-missing --audit
dataset: train_v1
split: component protocol splits
seed: component manifests
protocol: score_fusion query_context
risk notes: Fixed alpha and temperature are design priors. Do not tune alpha or temperature from P3 outer/final results. The mixed P2 aggregate remains diagnostic-only, while per-fixed-crop P2 assemblies now pass candidate audit.

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

## Protocol Evidence

Runs assembled locally on 2026-05-22 from genuine component score matrices
and source manifests pulled from A/C remote evidence packages. Checkpoints and
raw temporary dumps were not copied or committed.

- P1 seed42:
  `outputs/pr49_local_20260522_2015_car_query_noea_mconn_context_p1/car_fbstcnet_as_query_noea_eps3e4_m_conn_srfnet_whitening_eps3e4_conformer_context_fusion`.
  - Audit: PASS.
  - exact BA: 0.7318421.
  - DEP mean BA: 0.6652704; HC mean BA: 0.7651280; DEP/HC: 0.8694890.
- P2:
  `outputs/pr49_local_20260522_2021_car_query_noea_mconn_context_p2/car_fbstcnet_as_query_noea_eps3e4_m_conn_srfnet_whitening_eps3e4_conformer_context_fusion`.
  - Audit: DIAGNOSTIC_ONLY. Candidate audit fails on mixed P2 crop provenance
    semantics, not missing component evidence.
  - exact BA: 0.7135795.
- P2 fixed-crop candidate repair:
  `outputs/pr49_fix_p2_crop_candidate_20260522_230728/p2__eval_crop*/car_fbstcnet_as_query_noea_eps3e4_m_conn_srfnet_whitening_eps3e4_conformer_context_fusion`.
  - Audit: PASS for crop1 through crop5 after assembling each normalized
    protocol job separately.
  - crop1 exact BA 0.7291667; DEP/HC 0.9166667.
  - crop2 exact BA 0.7291667; DEP/HC 0.9166667.
  - crop3 exact BA 0.7916667; DEP/HC 1.0400000.
  - crop4 exact BA 0.7916667; DEP/HC 0.9230769.
  - crop5 exact BA 0.6875000; DEP/HC 0.8695652.
- P3 nested:
  `outputs/pr49_local_20260522_2124_car_query_noea_mconn_context_p3/car_fbstcnet_as_query_noea_eps3e4_m_conn_srfnet_whitening_eps3e4_conformer_context_fusion`.
  - Audit: PASS.
  - exact BA: 0.7365558.
  - DEP mean BA: 0.6574673; HC mean BA: 0.7761001; DEP/HC: 0.8471425.
  - Crop means: crop0 0.7108, crop1 0.7350, crop2 0.7467,
    crop3 0.7150, crop4 0.7542.

Interpretation: the route passes the lightweight gate and P3 candidate audit,
with DEP/HC comfortably above the 0.82 continuation threshold. It is very close
to the PR46 sibling P3 result but does not clearly supersede it, so keep it as
a CAR-query context sibling and fusion/balance reference rather than a new
primary route.
