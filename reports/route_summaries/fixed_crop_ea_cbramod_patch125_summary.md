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
audit_report_path: reports/route_summaries/evidence/fixed_crop_ea_cbramod_patch125_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_ea_cbramod_patch125_manifest.json
manifest_sha256: 3e6a3d16b0779bb94b8f4caaf6d2edc18847e33111b9e0e6cbd2e59125a987a3
