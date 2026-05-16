# fixed_crop_ea_deformer_dropout015 Summary

route_id: fixed_crop_ea_deformer_dropout015
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6364896
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_deformer_dropout015.yaml --run outputs/fixed_crop_ea_deformer_dropout015/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: generated draft; review before committing.
audit_report_path: reports/route_summaries/evidence/fixed_crop_ea_deformer_dropout015_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_ea_deformer_dropout015_manifest.json
manifest_sha256: 957d0ccde61e12ea5e02f92247c161b9b3853c1947ba937efe643b015cece4df
