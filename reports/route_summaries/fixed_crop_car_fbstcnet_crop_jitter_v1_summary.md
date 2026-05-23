# fixed_crop_car_fbstcnet_crop_jitter_v1 Summary

route_id: fixed_crop_car_fbstcnet_crop_jitter_v1
route_status: IDEA
audit_decision: WARN
gate: smoke
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.710519
decision: DO_NOT_PROMOTE_CURRENT_MAIN_ROUTE
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_car_fbstcnet_crop_jitter_v1.yaml --run outputs/protocol_runs/non_destructive_robustness_20260523_212655/p1_seed42_3fold/job_runs/p1__fixed_crop_car_fbstcnet_crop_jitter_v1__seed42__fold0 --gate smoke
dataset: train_v1
split: p1_seed42_3fold and p2_crop_stress
seed: 42
protocol: p1_repeated_group_kfold plus p2_pseudo_public_holdout
risk notes: Train-only deterministic crop jitter improved P2 crop1-5 mean to 0.758333, but P2 worst stayed at the baseline 0.354167 and P1 seed42 3-fold mean was 0.710519, below the baseline mean 0.757259 by -0.046740. Treat as crop-stress diagnostic only.
