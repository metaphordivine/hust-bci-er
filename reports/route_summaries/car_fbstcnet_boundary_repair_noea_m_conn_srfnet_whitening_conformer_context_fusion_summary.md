# car_fbstcnet_boundary_repair_noea_m_conn_srfnet_whitening_conformer_context_fusion Summary

route_id: car_fbstcnet_boundary_repair_noea_m_conn_srfnet_whitening_conformer_context_fusion
route_status: IDEA
audit_decision: PASS_P1_AND_P2_FIXED_CROPS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.7305856
decision: KEEP_FOR_P3_IF_COMPONENT_PROVENANCE_REMAINS_TRUSTED
reproduce: python scripts/run_score_fusion_routes.py --route-filter car_fbstcnet_boundary_repair_noea_m_conn_srfnet_whitening_conformer_context_fusion --export-missing --audit
dataset: train_v1
split: component protocol splits
seed: component manifests
protocol: score_fusion margin_adaptive_query_context
risk notes: P2 aggregate screens must not mix crop policies inside one candidate manifest. Use per-protocol-job P2 assemblies for candidate crop-provenance checks.

## Route Family

This route keeps the PR49 CAR FBSTCNet query and limits no-EA m-conn FBSTCNet,
SRFNet whitening eps3e4, and Conformer to boundary repair context. It is a
low-blast-radius DEP/HC balance check rather than a new backbone family.

## Protocol Evidence

Runs assembled locally on 2026-05-22 from genuine component score matrices
and source manifests already used for PR49-style score fusion. Checkpoints and
raw temporary dumps were not copied or committed.

- P1 seed42:
  `outputs/pr51_local_20260522_2252_boundary_p1/car_fbstcnet_boundary_repair_noea_m_conn_srfnet_whitening_conformer_context_fusion`.
  - Audit: PASS after this summary is present.
  - exact BA: 0.7305856.
  - DEP mean BA: 0.6656816; HC mean BA: 0.7630376; DEP/HC: 0.8724100.
- P2 aggregate screen:
  `outputs/pr51_local_20260522_2252_boundary_p2/car_fbstcnet_boundary_repair_noea_m_conn_srfnet_whitening_conformer_context_fusion`.
  - Audit: DIAGNOSTIC_ONLY. The original aggregate run mixed P2 crop policies
    in one manifest.
  - exact BA: 0.7139271.
  - DEP mean BA: 0.6523825; HC mean BA: 0.7446994; DEP/HC: 0.8761207.
- P2 fixed-crop candidate repair:
  `outputs/pr51_boundary_fix_p2_crop_candidate_20260522_230728/p2__eval_crop*/car_fbstcnet_boundary_repair_noea_m_conn_srfnet_whitening_conformer_context_fusion`.
  - Audit: PASS for crop1 through crop5 after assembling each normalized
    protocol job separately.
  - crop1 exact BA 0.7083333; DEP/HC 0.9565217.
  - crop2 exact BA 0.7291667; DEP/HC 0.9166667.
  - crop3 exact BA 0.7916667; DEP/HC 1.0400000.
  - crop4 exact BA 0.7916667; DEP/HC 0.9230769.
  - crop5 exact BA 0.6875000; DEP/HC 0.8695652.

Interpretation: the route passes the P1 lightweight gate and maintains DEP/HC
above 0.82. The aggregate P2 value is useful for stress comparison only;
fixed-crop P2 candidate evidence should use the per-job repaired runs.
