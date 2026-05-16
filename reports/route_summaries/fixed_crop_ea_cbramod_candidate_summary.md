# fixed_crop_ea_cbramod_candidate Summary

route_id: fixed_crop_ea_cbramod_candidate
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6181695999999999
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_cbramod_candidate.yaml --run outputs/fixed_crop_ea_cbramod_candidate/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: generated draft; review before committing.
audit_report_path: outputs/fixed_crop_ea_cbramod_candidate/candidate_20260516_seed42/audit_report.json
manifest_path: outputs/fixed_crop_ea_cbramod_candidate/candidate_20260516_seed42/manifest.json
manifest_sha256: 68b76fa3b986eee3a51dfbcedd06e1969dbf50faf03ccae4c6218b49755ebe2f
