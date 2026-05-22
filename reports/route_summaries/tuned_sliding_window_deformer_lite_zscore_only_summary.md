# tuned_sliding_window_deformer_lite_zscore_only Summary

route_id: tuned_sliding_window_deformer_lite_zscore_only
route_status: IDEA
audit_decision: PENDING_REMOTE_PROTOCOL_RUNS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: TBD
decision: HOLD
reproduce: python scripts/run_evaluation_protocol.py --protocol p3 --route-config configs/routes/models/tuned_sliding_window_deformer_lite_zscore_only.yaml --param-grid tuned_sliding_window_deformer_lite_zscore_only=configs/search/spaces/sliding_deformer_lite_window_shape.yaml --execute --execute-gate candidate
dataset: train_v1
split: P1/P2/P3 protocol runner
seed: protocol default unless specified by run manifest
protocol: no-EA sliding-window Deformer-lite ablation
risk notes: Placeholder summary for the no-EA Deformer-lite sliding-window ablation. Treat results as diagnostic for EA and window-shape effects only until matched fixed/sliding and P3 outer-test evidence are available.
