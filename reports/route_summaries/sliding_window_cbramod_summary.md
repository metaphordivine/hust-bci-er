# sliding_window_cbramod Summary

route_id: sliding_window_cbramod
route_status: CANDIDATE
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.5068525333333334
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route D:\hust-bci-er\configs\routes\models\sliding_window_cbramod.yaml --run outputs/candidate_all_models/20260516_132255/sliding_window_cbramod --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: candidate audit passed; compact manifest/audit evidence is stored under reports.
audit_report_path: reports/route_summaries/evidence/sliding_window_cbramod_audit_report.json
manifest_path: reports/route_summaries/evidence/sliding_window_cbramod_manifest.json
manifest_sha256: 1d473ef34e03af8ff283b9e6fddbff6f7edd9fe28a4623edf99cc8831aefe753