# fixed_crop_ea_fbstcnet_c_only Summary

route_id: fixed_crop_ea_fbstcnet_c_only
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.7021090666666666
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_fbstcnet_c_only.yaml --run outputs/fixed_crop_ea_fbstcnet_c_only/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: generated draft; review before committing.
audit_report_path: reports/route_summaries/evidence/fixed_crop_ea_fbstcnet_c_only_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_ea_fbstcnet_c_only_manifest.json
manifest_sha256: b2e04528c08b8f9f75a02cb75b5ac84facdcadf432783382d80d9010e6850458
