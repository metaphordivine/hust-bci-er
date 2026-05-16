# fixed_crop_ea_cbramod_patch500 Summary

route_id: fixed_crop_ea_cbramod_patch500
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6178485333333333
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_cbramod_patch500.yaml --run outputs/fixed_crop_ea_cbramod_patch500/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: generated draft; review before committing.
audit_report_path: outputs/fixed_crop_ea_cbramod_patch500/candidate_20260516_seed42/audit_report.json
manifest_path: outputs/fixed_crop_ea_cbramod_patch500/candidate_20260516_seed42/manifest.json
manifest_sha256: 23263576920a24371fb77b927b6e292ee575c296f0b64a9d3f3ed5f8c2de37c3
