# fixed_crop_car_fbstcnet_region_scale_down_weak_v1 Summary

route_id: fixed_crop_car_fbstcnet_region_scale_down_weak_v1
route_status: IDEA
audit_decision: PENDING
gate: smoke
primary_metric: exact_single_crop_expected_BA
decision: PENDING_REMOTE_EVIDENCE
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_car_fbstcnet_region_scale_down_weak_v1.yaml --run outputs/protocol_runs/<run_root>/p1/job_runs/<job_id> --gate smoke
dataset: train_v1
split: P1full and P2 planned
seed: 42
protocol: p1_repeated_group_kfold plus p2_pseudo_public_holdout
risk notes: Augmentation must remain train-only. First version excludes frontal region to avoid repeating frontal artifact manipulation risk.
