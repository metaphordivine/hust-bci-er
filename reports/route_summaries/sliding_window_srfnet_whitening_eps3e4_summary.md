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

## Latest Remote Evidence

Backfill and PR35-aligned runs on 2026-05-22 produced:

- P1 seed42 mean BA: 0.6328.
- P2 mean BA: 0.6449.
- P3 final mean BA: 0.6294 from `outputs/backfill_pr35_A_20260522_160445_sliding_srfnet_whitening_eps3e4_p3`.
- P1 5x5 completion backfill:
  - seeds 123/456 mean BA: 0.6044 from `outputs/backfill_pr35_A_20260522_1900_sliding_srfnet_whitening_eps3e4_p1_seeds123_456`.
  - seed 789 mean BA: 0.6088 from `outputs/backfill_pr35_B_20260522_1900_sliding_srfnet_whitening_eps3e4_p1_seed789`.
  - seed 1024 mean BA: 0.5600 from `outputs/backfill_pr35_C_20260522_1900_sliding_srfnet_whitening_eps3e4_p1_seed1024`.
  - Approximate full P1 5x5 mean BA: 0.6021, combining the prior seed42 aggregate with the four backfilled seeds.

Interpretation: the route is not competitive as a standalone model. Preserve
it only as a balance/component diagnostic for score fusion and sliding-window
ablation, especially if DEP/HC balance remains complementary to stronger
FBSTCNet or DGCNN routes.
