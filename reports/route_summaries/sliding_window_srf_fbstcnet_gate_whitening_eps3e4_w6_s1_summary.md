# sliding_window_srf_fbstcnet_gate_whitening_eps3e4_w6_s1 Summary

route_id: sliding_window_srf_fbstcnet_gate_whitening_eps3e4_w6_s1
route_status: IDEA
audit_decision: PENDING_LIGHTWEIGHT_SCREEN
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: TBD
decision: HOLD
reproduce: python scripts/run_evaluation_protocol.py --protocol p1 --route-config configs/routes/models/sliding_window_srf_fbstcnet_gate_whitening_eps3e4_w6_s1.yaml --seed 42 --n-folds 5 --execute --execute-gate candidate
dataset: train_v1
split: P1 seed42 and P2 protocol runner
seed: 42
protocol: sliding-window SRFNet and FBSTCNet gated-backbone IDEA screen
risk notes: Placeholder summary for a new backbone IDEA route. Do not promote from smoke or a single split. Run P1 seed42 and P2 first; only continue to full experiments if it passes the user-defined lightweight promotion gate.
