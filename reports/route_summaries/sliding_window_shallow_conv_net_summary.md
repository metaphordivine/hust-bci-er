# sliding_window_shallow_conv_net Summary

route_id: sliding_window_shallow_conv_net
route_status: CANDIDATE
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.5338674666666666
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route D:\hust-bci-er\configs\routes\models\sliding_window_shallow_conv_net.yaml --run outputs/candidate_all_models/20260516_132255/sliding_window_shallow_conv_net --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: candidate audit passed; compact manifest/audit evidence is stored under reports.
audit_report_path: reports/route_summaries/evidence/sliding_window_shallow_conv_net_audit_report.json
manifest_path: reports/route_summaries/evidence/sliding_window_shallow_conv_net_manifest.json
manifest_sha256: 1c5e42cfacf4951e2579b17c17c263682def382bd6cfea5ce645df387133b89f