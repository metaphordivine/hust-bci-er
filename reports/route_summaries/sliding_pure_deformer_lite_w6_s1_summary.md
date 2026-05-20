# sliding_pure_deformer_lite_w6_s1 Summary

route_id: sliding_pure_deformer_lite_w6_s1
route_status: IDEA
audit_decision: PENDING_REAL_RUN
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: pending
decision: HOLD
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/sliding_pure_deformer_lite_w6_s1.yaml --run outputs/<run_dir>/sliding_pure_deformer_lite_w6_s1 --gate candidate
dataset: train_v1
split: pending
seed: 42
protocol: p1_repeated_group_kfold
risk notes: Sliding-window Deformer-lite family-diversity control. Status remains IDEA until P1/P2/P3 candidate runs finish and audit evidence passes.
