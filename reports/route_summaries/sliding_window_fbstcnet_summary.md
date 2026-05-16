# sliding_window_fbstcnet Summary

route_id: sliding_window_fbstcnet
route_status: CANDIDATE
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6028840000000001
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/sliding_window_fbstcnet.yaml --run outputs/candidate_all_models/20260516_132255/sliding_window_fbstcnet --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: candidate audit passed; compact manifest/audit evidence is stored under reports.
audit_report_path: reports/route_summaries/evidence/sliding_window_fbstcnet_audit_report.json
manifest_path: reports/route_summaries/evidence/sliding_window_fbstcnet_manifest.json
manifest_sha256: 7c126bced6bd824d4b492d37125cb17fbec79cc19695bf223684283fdacd7d4d