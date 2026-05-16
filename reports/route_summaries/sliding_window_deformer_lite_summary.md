# sliding_window_deformer_lite Summary

route_id: sliding_window_deformer_lite
route_status: CANDIDATE
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.5794466666666666
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/sliding_window_deformer_lite.yaml --run outputs/candidate_all_models/20260516_132255/sliding_window_deformer_lite --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: candidate audit passed; compact manifest/audit evidence is stored under reports.
audit_report_path: reports/route_summaries/evidence/sliding_window_deformer_lite_audit_report.json
manifest_path: reports/route_summaries/evidence/sliding_window_deformer_lite_manifest.json
manifest_sha256: de5801f3f982bd815c3e3d4b62fb8927eaabd907cec73c0cd599d21c2c716aa9