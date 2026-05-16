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
audit_report_path: reports/route_summaries/evidence/fixed_crop_ea_cbramod_candidate_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_ea_cbramod_candidate_manifest.json
manifest_sha256: cd660cad9ece2652e71e6156e8e8a901b7d88d7052109447cf4c6db06a3c4fc8
