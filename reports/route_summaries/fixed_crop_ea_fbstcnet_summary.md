# fixed_crop_ea_fbstcnet Summary

route_id: fixed_crop_ea_fbstcnet
route_status: CANDIDATE
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.686144
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_fbstcnet.yaml --run outputs/fixed_crop_ea_fbstcnet/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: generated draft; review before committing.
audit_report_path: outputs/fixed_crop_ea_fbstcnet/candidate_20260516_seed42/audit_report.json
manifest_path: outputs/fixed_crop_ea_fbstcnet/candidate_20260516_seed42/manifest.json
manifest_sha256: a15e940d811298d4c6039bf450f2e16defcfadc261de7f8750efe5046615b468
