# sliding_ea_fbstcnet_m_conn_w6_s1 Summary

route_id: sliding_ea_fbstcnet_m_conn_w6_s1
route_status: IDEA
audit_decision: PENDING_REAL_RUN
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: pending
decision: HOLD
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/sliding_ea_fbstcnet_m_conn_w6_s1.yaml --run outputs/<run_dir>/sliding_ea_fbstcnet_m_conn_w6_s1 --gate candidate
dataset: train_v1
split: pending
seed: 42
protocol: p1_repeated_group_kfold
risk notes: 6.0 s / 1.0 s sliding-window baseline for the best fixed-crop P3 FBSTCNet-M connection-light route. Status remains IDEA until candidate evidence passes.
