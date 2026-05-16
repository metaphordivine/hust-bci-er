# sliding_window_conformer_lite Summary

route_id: sliding_window_conformer_lite
route_status: CANDIDATE
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.5866986666666667
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/sliding_window_conformer_lite.yaml --run outputs/candidate_all_models/20260516_132255/sliding_window_conformer_lite --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: candidate audit passed; compact manifest/audit evidence is stored under reports.
audit_report_path: reports/route_summaries/evidence/sliding_window_conformer_lite_audit_report.json
manifest_path: reports/route_summaries/evidence/sliding_window_conformer_lite_manifest.json
manifest_sha256: d37798691064f590815025cb33ac49bdb095046fb56d54a7c13169c661c9d196