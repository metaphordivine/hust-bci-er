# fixed_crop_ea_whitening_eps1e3_fbstcnet_m_conn_light Summary

route_id: fixed_crop_ea_whitening_eps1e3_fbstcnet_m_conn_light
route_status: IDEA
audit_decision: BLOCKED
gate:
primary_metric: exact_single_crop_expected_BA
decision: pending real evidence
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_whitening_eps1e3_fbstcnet_m_conn_light.yaml --gate smoke
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: design-only route config; run P2 crop-stress, then P3 nested selection, then P1 3x5 only if the route stays competitive.

## Route Family

This route is part of the EA + whitening + FBSTCNet branch stability grid. The
family tests whether subject alignment, whitening regularization, FBSTCNet
branch choice, or the connectivity window explains the strong fixed-crop P3
results without changing split definitions or evaluation protocol semantics.
This is the lead design summary for the grid; sibling routes stay summary-light
until a real run produces candidate-grade evidence.

## Candidate Set

- `fixed_crop_ea_whitening_eps1e3_fbstcnet`: existing baseline, mixed branch,
  `gamma=300`.
- `fixed_crop_ea_whitening_eps1e3_fbstcnet_m_conn_light`: mixed branch,
  shorter connectivity window, `gamma=200`.
- `fixed_crop_ea_whitening_eps1e3_fbstcnet_m_power_light`: mixed branch,
  stronger power dropout and coarser power pooling.
- `fixed_crop_ea_whitening_eps1e3_fbstcnet_p_only`: power branch only.
- `fixed_crop_ea_whitening_eps1e3_fbstcnet_c_only`: connectivity branch only.
- `fixed_crop_ea_whitening_eps3e4_fbstcnet`: epsilon sensitivity check.

The matching P3 search space is
`configs/search/spaces/fixed_crop_ea_whitening_fbstcnet_branch_grid.yaml`.

## Interpretability Checks

- Compare crop-policy spread and worst-crop BA before promotion.
- Compare per-fold branch stability: mixed, power-only, and connectivity-only.
- Compare whitening eps1e3 vs eps3e4 as sensitivity, not as a protocol change.
- Inspect hard-subject deltas against the existing zscore FBSTCNet branch runs.
