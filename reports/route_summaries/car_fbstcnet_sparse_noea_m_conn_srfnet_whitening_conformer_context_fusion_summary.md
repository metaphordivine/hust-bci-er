# car_fbstcnet_sparse_noea_m_conn_srfnet_whitening_conformer_context_fusion Summary

route_id: car_fbstcnet_sparse_noea_m_conn_srfnet_whitening_conformer_context_fusion
route_status: IDEA
audit_decision: PASS_P1_AND_P2_FIXED_CROPS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.7310309333333334
decision: KEEP_FOR_P3_IF_COMPONENT_PROVENANCE_REMAINS_TRUSTED
reproduce: python scripts/run_score_fusion_routes.py --route-filter car_fbstcnet_sparse_noea_m_conn_srfnet_whitening_conformer_context_fusion --export-missing --audit
dataset: train_v1
split: component protocol splits
seed: component manifests
protocol: score_fusion sparse query_context
risk notes: The mixed P2 aggregate remains diagnostic-only, while per-fixed-crop P2 assemblies pass candidate audit. Do not tune alpha or temperature from P2/P3 outer results.

## Route Family

This route is the low-alpha sparse query-context reweight of the PR49 component
stack. Fixed-crop CAR FBSTCNet remains the query while no-EA eps3e4 m-conn
FBSTCNet, SRFNet whitening eps3e4, and Conformer are restricted to context.

## Protocol Evidence

Runs assembled locally on 2026-05-22 from genuine component score matrices
and source manifests already used for PR49-style score fusion. Checkpoints and
raw temporary dumps were not copied or committed.

- P1 seed42:
  `outputs/pr51_local_20260522_2252_sparse_p1/car_fbstcnet_sparse_noea_m_conn_srfnet_whitening_conformer_context_fusion`.
  - Audit: PASS after this summary is present.
  - exact BA: 0.7310309.
  - DEP mean BA: 0.6642408; HC mean BA: 0.7644260; DEP/HC: 0.8689406.
- P2:
  `outputs/pr51_local_20260522_2252_sparse_p2/car_fbstcnet_sparse_noea_m_conn_srfnet_whitening_conformer_context_fusion`.
  - Audit: DIAGNOSTIC_ONLY. Candidate audit fails on P2 crop provenance
    semantics, not missing or synthetic component evidence.
  - exact BA: 0.7109108.
  - DEP mean BA: 0.6518317; HC mean BA: 0.7404503; DEP/HC: 0.8803184.
- P2 fixed-crop candidate repair:
  `outputs/pr51_sparse_fix_p2_crop_candidate_20260522_230728/p2__eval_crop*/car_fbstcnet_sparse_noea_m_conn_srfnet_whitening_conformer_context_fusion`.
  - Audit: PASS for crop1 through crop5 after assembling each normalized
    protocol job separately.
  - crop1 exact BA 0.7291667; DEP/HC 0.9166667.
  - crop2 exact BA 0.7291667; DEP/HC 0.9166667.
  - crop3 exact BA 0.7916667; DEP/HC 1.0400000.
  - crop4 exact BA 0.7708333; DEP/HC 0.9600000.
  - crop5 exact BA 0.6875000; DEP/HC 0.8695652.

Interpretation: the route passes the P1 lightweight gate and maintains DEP/HC
comfortably above 0.82. The aggregate P2 value is useful for stress comparison
only; fixed-crop P2 candidate evidence should use the per-job repaired runs.
