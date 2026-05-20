# fixed_crop_ea_dgcnn Summary

route_id: fixed_crop_ea_dgcnn
route_status: IDEA
audit_decision: PENDING_REAUDIT
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6931897173333333
decision: HOLD
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_dgcnn.yaml --run outputs/protocol_p1_full_lane_pr15_models/fixed_crop_ea_dgcnn/job_runs/p1__fixed_crop_ea_dgcnn__seed42__fold0 --gate candidate
dataset: train_v1
split: p1 repeated group-kfold, 5 seeds x 5 folds
seed: 42, 123, 456, 789, 1024
protocol: p1_repeated_group_kfold
completed_jobs: 25
aggregate_metric_mean: 0.6931897173333333
aggregate_metric_min: 0.6233333333333333
aggregate_metric_max: 0.7426714666666666
risk notes: P1 full artifacts completed for all 25 jobs, but the first remote audit pass was blocked only because this route summary was missing. Rerun candidate audit before treating this as PASS; route status remains IDEA until promotion is explicitly reviewed.
