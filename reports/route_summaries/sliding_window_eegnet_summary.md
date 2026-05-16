# sliding_window_eegnet Summary

route_id: sliding_window_eegnet
route_status: CANDIDATE
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.637912
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/sliding_window_eegnet.yaml --run outputs/candidate_all_models/20260516_132255/sliding_window_eegnet --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: candidate audit passed; compact manifest/audit evidence is stored under reports.
audit_report_path: reports/route_summaries/evidence/sliding_window_eegnet_audit_report.json
manifest_path: reports/route_summaries/evidence/sliding_window_eegnet_manifest.json
manifest_sha256: d3d64d46ccc508ab6c84afbfbbca139cc3b7a04ca3df23a7a482a8fa0424c030