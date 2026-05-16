# sliding_window_srfnet Summary

route_id: sliding_window_srfnet
route_status: CANDIDATE
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6596266666666667
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route D:\hust-bci-er\configs\routes\models\sliding_window_srfnet.yaml --run outputs/candidate_all_models/20260516_132255/sliding_window_srfnet --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: candidate audit passed; compact manifest/audit evidence is stored under reports.
audit_report_path: reports/route_summaries/evidence/sliding_window_srfnet_audit_report.json
manifest_path: reports/route_summaries/evidence/sliding_window_srfnet_manifest.json
manifest_sha256: d3e70e2b1fbe9a4861b1de74a748784ec7030d5ca505cfe8bcf7cc8e95619124