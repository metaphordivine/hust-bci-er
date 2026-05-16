# fixed_crop_ea_cbramod_patch125 Summary

route_id: fixed_crop_ea_cbramod_patch125
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6218146666666667
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_cbramod_patch125.yaml --run outputs/fixed_crop_ea_cbramod_patch125/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: generated draft; review before committing.
audit_report_path: outputs/fixed_crop_ea_cbramod_patch125/candidate_20260516_seed42/audit_report.json
manifest_path: outputs/fixed_crop_ea_cbramod_patch125/candidate_20260516_seed42/manifest.json
manifest_sha256: 5f0e3e7ab3c737f22d9c212a5d41f718887bc6c76e47718212c7fd706e6fa4fc
