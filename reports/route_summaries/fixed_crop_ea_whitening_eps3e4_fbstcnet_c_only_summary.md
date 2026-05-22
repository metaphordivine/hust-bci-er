# fixed_crop_ea_whitening_eps3e4_fbstcnet_c_only Summary

route_id: fixed_crop_ea_whitening_eps3e4_fbstcnet_c_only
route_status: IDEA
audit_decision: BLOCKED
gate:
primary_metric: exact_single_crop_expected_BA
decision: pending real evidence
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_whitening_eps3e4_fbstcnet_c_only.yaml --gate smoke
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: Connectivity-only branch control for eps3e4 whitening. Stop as standalone if P2 crop stress stays below the mixed-branch route.

## Route Family

This route tests whether the connectivity branch alone explains any eps3e4
whitening gain under Euclidean alignment. It preserves the same split-first
fixed-crop aggregation and candidate audit rules as the FBSTCNet whitening grid.

## Promotion Checks

- Run P1 seed42 and P2.
- Compare against mixed and power-only branch controls.
- Use mainly for branch attribution or fusion diversity unless it is stable across crop policies.
