# tuned_sliding_window_fbstcnet_zscore_only Summary

route_id: tuned_sliding_window_fbstcnet_zscore_only
route_status: IDEA
audit_decision: PENDING_REMOTE_PROTOCOL_RUNS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: TBD
decision: HOLD
reproduce: python scripts/run_evaluation_protocol.py --protocol p3 --route-config configs/routes/models/tuned_sliding_window_fbstcnet_zscore_only.yaml --param-grid tuned_sliding_window_fbstcnet_zscore_only=configs/search/spaces/sliding_fbstcnet_window_shape.yaml --execute --execute-gate candidate
dataset: train_v1
split: P1/P2/P3 protocol runner
seed: protocol default unless specified by run manifest
protocol: no-EA sliding-window FBSTCNet ablation
risk notes: Placeholder summary for the no-EA sliding-window FBSTCNet ablation. This route is intended only for matched EA and sliding-window diagnosis until P1/P2/P3 evidence is reviewed; do not promote from a single split or P2-only result.
