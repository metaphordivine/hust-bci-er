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
audit_report_path: outputs/fixed_crop_ea_fbstcnet_c_only/candidate_20260516_seed42/audit_report.json
manifest_path: outputs/fixed_crop_ea_fbstcnet_c_only/candidate_20260516_seed42/manifest.json
manifest_sha256: b7c88265d0566a3c17a8ca01e007c410f324a503d11cd060ef978b9fabfb915f
