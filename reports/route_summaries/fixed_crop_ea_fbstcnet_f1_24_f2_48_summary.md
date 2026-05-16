# fixed_crop_ea_fbstcnet_f1_24_f2_48 Summary

route_id: fixed_crop_ea_fbstcnet_f1_24_f2_48
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6649416
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_fbstcnet_f1_24_f2_48.yaml --run outputs/fixed_crop_ea_fbstcnet_f1_24_f2_48/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: generated draft; review before committing.
audit_report_path: reports/route_summaries/evidence/fixed_crop_ea_fbstcnet_f1_24_f2_48_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_ea_fbstcnet_f1_24_f2_48_manifest.json
manifest_sha256: d02d8f6bb3bd39cab799a61f6775ea25c6f6d2b3c22e5f04a40986ab2d13c740
