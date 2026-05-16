# ea_deformer Summary

route_id: ea_deformer
route_status: CANDIDATE
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6162066666666667
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route D:\hust-bci-er\configs\routes\models\ea_deformer.yaml --run outputs/candidate_all_models/20260516_132255/ea_deformer --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: candidate audit passed; compact manifest/audit evidence is stored under reports.
audit_report_path: reports/route_summaries/evidence/ea_deformer_audit_report.json
manifest_path: reports/route_summaries/evidence/ea_deformer_manifest.json
manifest_sha256: be7d0a07b4380ee0e267cccea4486f7402c24d3e11f5350f17b85efe90e686e2
