# fixed_crop_deformer_lite_zscore_tuned Summary

route_id: fixed_crop_deformer_lite_zscore_tuned
route_status: IDEA
audit_decision: PENDING_REMOTE_PROTOCOL_RUNS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: TBD
decision: HOLD
reproduce: python scripts/run_evaluation_protocol.py --protocol p2 --route-config configs/routes/models/fixed_crop_deformer_lite_zscore_tuned.yaml --execute --execute-gate candidate
dataset: train_v1
split: P1/P2/P3 protocol runner
seed: protocol default unless specified by run manifest
protocol: fixed-crop no-EA Deformer-lite ablation
risk notes: Placeholder summary for the fixed-crop Deformer-lite zscore baseline used to isolate sliding-window effects. Do not compare against FBSTCNet without controlling preprocessing, window policy, and protocol.
