# fixed_crop_car_fbstcnet_time_shift_aug_light_v1 Summary

route_id: fixed_crop_car_fbstcnet_time_shift_aug_light_v1
route_status: IDEA
audit_decision: PENDING
gate: smoke
primary_metric: exact_single_crop_expected_BA
decision: PENDING_REMOTE_EVIDENCE
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_car_fbstcnet_time_shift_aug_light_v1.yaml --run outputs/protocol_runs/<run_root>/p1full_seed42_5fold/job_runs/<job_id> --gate smoke
dataset: train_v1
split: p1full_seed42_5fold and p2_crop_stress planned
seed: 42
protocol: p1_repeated_group_kfold plus p2_pseudo_public_holdout
risk notes: Combined augmentation route. All transforms must remain train-only; validation/test fixed crops are unchanged.
