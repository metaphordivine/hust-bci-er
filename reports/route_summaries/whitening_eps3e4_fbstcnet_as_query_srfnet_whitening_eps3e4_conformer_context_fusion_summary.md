# whitening_eps3e4_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion Summary

route_id: whitening_eps3e4_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion
route_status: IDEA
audit_decision: BLOCKED
gate:
primary_metric: exact_single_crop_expected_BA
decision: pending genuine component evidence
reproduce: python scripts/run_score_fusion_routes.py --route-filter whitening_eps3e4_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion --export-missing --audit
dataset: train_v1
split: component protocol splits
seed: component manifests
protocol: score_fusion query_context
risk notes: Uses only genuine component score matrices. Do not tune alpha or temperature from outer-test results.

## Route Family

This score-fusion route uses the no-EA eps3e4 FBSTCNet component as query and
adds SRFNet whitening eps3e4 plus Conformer as context components. It is a
paired fusion test for whether no-EA whitening evidence complements the current
CAR and EA score-fusion family.

## Promotion Checks

- Require genuine source manifests for all components.
- Run dry-run/component binding checks before assembly.
- Treat missing component evidence as a blocker, not as zero-weight evidence.
