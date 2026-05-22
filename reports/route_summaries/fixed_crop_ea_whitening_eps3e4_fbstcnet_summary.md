# fixed_crop_ea_whitening_eps3e4_fbstcnet Summary

route_id: fixed_crop_ea_whitening_eps3e4_fbstcnet
route_status: IDEA
audit_decision: PENDING_REMOTE_PROTOCOL_RUNS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: TBD
decision: HOLD
reproduce: python scripts/run_evaluation_protocol.py --protocol p2 --route-config configs/routes/models/fixed_crop_ea_whitening_eps3e4_fbstcnet.yaml --execute --execute-gate candidate
dataset: train_v1
split: P1/P2/P3 protocol runner
seed: protocol default unless specified by run manifest
protocol: EA plus whitening eps3e4 FBSTCNet ablation
risk notes: Placeholder summary for the eps3e4 whitening route with Euclidean alignment. Interpret only as an interaction test against the no-EA eps3e4 and eps1e3 whitening controls; do not promote from P2 alone.
