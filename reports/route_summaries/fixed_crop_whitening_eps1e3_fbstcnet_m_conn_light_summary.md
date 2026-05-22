# fixed_crop_whitening_eps1e3_fbstcnet_m_conn_light Summary

route_id: fixed_crop_whitening_eps1e3_fbstcnet_m_conn_light
route_status: IDEA
audit_decision: BLOCKED
gate:
primary_metric: exact_single_crop_expected_BA
decision: pending real evidence
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_whitening_eps1e3_fbstcnet_m_conn_light.yaml --gate smoke
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: No-EA paired control for the EA whitening branch-stability grid. Treat as a diagnostic unless P1 seed42 or P2 crop stress approaches the current high-score threshold.

## Route Family

This route removes Euclidean alignment from the eps1e3 whitening FBSTCNet
mixed-branch configuration while keeping the shorter connectivity window
`gamma=200`. It is intended to separate whitening and branch effects from EA
effects without changing splits, evaluation protocols, or model inputs.

## Promotion Checks

- Run P1 seed42 and P2 before any broader P1/P3 expansion.
- Compare against the EA sibling `fixed_crop_ea_whitening_eps1e3_fbstcnet_m_conn_light`.
- Keep only as fusion/component evidence if P2 mean or DEP/HC balance does not meet the exploration threshold.
