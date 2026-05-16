# fixed_crop_fbstcnet_zscore_only Summary

route_id: fixed_crop_fbstcnet_zscore_only
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6957770666666666
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_fbstcnet_zscore_only.yaml --run outputs/fixed_crop_fbstcnet_zscore_only/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: generated draft; review before committing.
audit_report_path: outputs/fixed_crop_fbstcnet_zscore_only/candidate_20260516_seed42/audit_report.json
manifest_path: outputs/fixed_crop_fbstcnet_zscore_only/candidate_20260516_seed42/manifest.json
manifest_sha256: 8ef6638e5ae0308c40c78212d08d88563a7b4e0a66c652ab11f03a2a6fa8d02e
