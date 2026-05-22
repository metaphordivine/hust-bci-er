# fixed_crop_ea_whitening_eps3e4_fbstcnet_m_conn_light Summary

route_id: fixed_crop_ea_whitening_eps3e4_fbstcnet_m_conn_light
route_status: IDEA
audit_decision: BLOCKED
gate:
primary_metric: exact_single_crop_expected_BA
decision: pending real evidence
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_whitening_eps3e4_fbstcnet_m_conn_light.yaml --gate smoke
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: EA eps3e4 branch-stability control. Advance only if it improves over the no-EA paired route or materially improves DEP/HC balance.

## Route Family

This route keeps Euclidean alignment and eps3e4 whitening, then uses the
FBSTCNet mixed branch with `gamma=200`. It is the EA sibling for the no-EA
eps3e4 m-conn-light control.

## Promotion Checks

- Run P1 seed42 and P2 before P3.
- Compare against `fixed_crop_whitening_eps3e4_fbstcnet_m_conn_light`.
- Inspect fold spread and DEP/HC balance before using it as a standalone route.
