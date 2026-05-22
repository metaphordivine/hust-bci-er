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

## Latest Remote Evidence

Remote PR44 runs on 2026-05-22 produced:

- P1 seed42 mean BA: 0.6857.
- P2 eval mean BA: 0.6429.
- P3 final mean BA: 0.7018 from `outputs/pr44_promote_A_20260522_152350_ea_eps3e4_conly_p3` and matching C-side replication.

Interpretation: the connectivity-only branch underperforms the mixed-branch
routes and should not be expanded as a standalone route. It remains useful for
branch attribution and fusion diversity diagnostics only.
