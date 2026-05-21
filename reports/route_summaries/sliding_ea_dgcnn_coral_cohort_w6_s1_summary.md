# sliding_ea_dgcnn_coral_cohort_w6_s1 Summary

route_id: sliding_ea_dgcnn_coral_cohort_w6_s1
route_status: IDEA
audit_decision: PENDING_REAL_RUN
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: pending
decision: HOLD
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/sliding_ea_dgcnn_coral_cohort_w6_s1.yaml --run outputs/<run_dir>/sliding_ea_dgcnn_coral_cohort_w6_s1 --gate candidate
dataset: train_v1
split: pending
seed: 42
protocol: p1_repeated_group_kfold
risk notes: Sliding-window DGCNN with train-side cohort CORAL feature alignment. This is intended as a lower-variance DEP/HC gap probe against DANN; route status remains IDEA until full P1/P2/P3 evidence passes candidate audit.
