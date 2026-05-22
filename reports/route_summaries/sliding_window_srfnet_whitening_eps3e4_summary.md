# sliding_window_srfnet_whitening_eps3e4 Summary

route_id: sliding_window_srfnet_whitening_eps3e4
route_status: IDEA
audit_decision: PENDING_REMOTE_PROTOCOL_RUNS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: TBD
decision: HOLD
reproduce: python scripts/run_evaluation_protocol.py --protocol p2 --route-config configs/routes/models/sliding_window_srfnet_whitening_eps3e4.yaml --execute --execute-gate candidate
dataset: train_v1
split: P1/P2/P3 protocol runner
seed: protocol default unless specified by run manifest
protocol: SRFNet whitening eps3e4 sliding-window score component
risk notes: Placeholder summary for the SRFNet eps3e4 component route. Its main purpose is score-fusion complementarity and sliding-window/whitening diagnosis; require leave-one-out fusion and matched protocol evidence before treating it as a standalone route.
