# fixed_crop_ea_deformer Summary

route_id: fixed_crop_ea_deformer
route_status: CANDIDATE
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6295301333333333
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_deformer.yaml --run outputs/model_route_exploration/20260516_candidate/fixed_crop_ea_deformer --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: candidate audit passed; improves over ea_deformer 0.616207 by +0.013323; compact evidence is stored under reports.
audit_report_path: reports/route_summaries/evidence/fixed_crop_ea_deformer_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_ea_deformer_manifest.json
manifest_sha256: 42ecae29b09369a0d8f36c141aee4b09a3e7709ee1ce360760ecf801d89fa54f
