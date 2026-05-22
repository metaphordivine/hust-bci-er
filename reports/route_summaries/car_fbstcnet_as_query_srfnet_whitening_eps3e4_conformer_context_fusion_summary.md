# car_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion Summary

route_id: car_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion
route_status: IDEA
audit_decision: PENDING_COMPONENT_PROTOCOL_RUNS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: TBD
decision: HOLD
reproduce: python scripts/run_score_fusion_routes.py --route-filter car_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion --base-runs-dir outputs/<component_protocol_runs> --export-missing --output-dir outputs/<score_fusion_run> --audit
dataset: train_v1
split: P1/P2 protocol runner component splits
seed: protocol default unless specified by component manifests
protocol: score_fusion query_context
risk notes: Placeholder summary only. This route reuses the existing query-context fusion method and replaces the EA FBSTCNet query with fixed-crop CAR FBSTCNet while adding SRFNet whitening_eps3e4 for DEP/HC balance. Candidate evidence requires genuine component score matrices and source manifests; do not tune weights from outer results.
