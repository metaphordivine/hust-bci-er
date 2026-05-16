# fixed_crop_ea_deformer_ff2 Summary

route_id: fixed_crop_ea_deformer_ff2
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6754189333333334
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_deformer_ff2.yaml --run outputs/fixed_crop_ea_deformer_ff2/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: generated draft; review before committing.
audit_report_path: reports/route_summaries/evidence/fixed_crop_ea_deformer_ff2_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_ea_deformer_ff2_manifest.json
manifest_sha256: d57c35f702043e7425e5a112537dc28e620a820d7b0395709f5798969d74dea7
