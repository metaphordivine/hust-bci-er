# sliding_ea_dgcnn_dann_lambda010_w8_s0p5 Summary

route_id: sliding_ea_dgcnn_dann_lambda010_w8_s0p5
route_status: IDEA
audit_decision: PENDING_REAL_RUN
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: pending
decision: HOLD
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/sliding_ea_dgcnn_dann_lambda010_w8_s0p5.yaml --run outputs/<run_dir>/sliding_ea_dgcnn_dann_lambda010_w8_s0p5 --gate candidate
dataset: train_v1
split: pending
seed: 42
protocol: p1_repeated_group_kfold
risk notes: Combines DANN lambda=0.1 with the P3-selected 8s/0.5s DGCNN window shape. Status remains IDEA until full P1/P2/P3 evidence passes candidate audit.
