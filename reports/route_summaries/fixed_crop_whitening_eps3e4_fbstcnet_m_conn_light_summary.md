# fixed_crop_whitening_eps3e4_fbstcnet_m_conn_light Summary

route_id: fixed_crop_whitening_eps3e4_fbstcnet_m_conn_light
route_status: IDEA
audit_decision: BLOCKED
gate:
primary_metric: exact_single_crop_expected_BA
decision: pending real evidence
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_whitening_eps3e4_fbstcnet_m_conn_light.yaml --gate smoke
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: No-EA eps3e4 paired control. Do not promote from a single P2 crop-policy result; require P1 seed42 and DEP/HC balance checks.

## Route Family

This route pairs eps3e4 whitening with the mixed FBSTCNet branch and
`gamma=200`, but omits Euclidean alignment. It directly tests whether the
observed eps3e4 signal is from whitening/branch settings rather than EA.

## Promotion Checks

- Run P1 seed42 and P2 as the lightweight exploration gate.
- Compare with `fixed_crop_ea_whitening_eps3e4_fbstcnet_m_conn_light`.
- If P2 stays below the current high-score threshold, keep only as a score-fusion component candidate.
