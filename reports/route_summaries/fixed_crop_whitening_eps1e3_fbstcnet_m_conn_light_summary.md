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

## Latest Remote Evidence

Remote PR44 runs on 2026-05-22 produced:

- P1 seed42 mean BA: 0.7108.
- P1 extra seeds 123/456 mean BA: 0.7060.
- P2 eval mean BA: 0.6518.
- P3 final mean BA: 0.7127 from `outputs/pr44_promote_A_20260522_152347_noea_eps1e3_mconn_p3` and matching C-side replication.

Interpretation: no-EA eps1e3 is close to no-EA eps3e4 and slightly ahead of
the EA eps3e4 sibling on P3, but its P2 score is too low for standalone
promotion. Keep it as an EA/whitening ablation and possible score-fusion
component.
