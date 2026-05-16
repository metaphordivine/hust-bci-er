# sparse_sliding_window_shallow_conv_net Summary

route_id: sparse_sliding_window_shallow_conv_net
route_status: CANDIDATE
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.5389253333333334
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/sparse_sliding_window_shallow_conv_net.yaml --run outputs/model_route_exploration/20260516_candidate/sparse_sliding_window_shallow_conv_net --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: candidate audit passed; improves over sliding_window_shallow_conv_net 0.533867 by +0.005058; compact evidence is stored under reports.
audit_report_path: reports/route_summaries/evidence/sparse_sliding_window_shallow_conv_net_audit_report.json
manifest_path: reports/route_summaries/evidence/sparse_sliding_window_shallow_conv_net_manifest.json
manifest_sha256: 1d41649a29d208c0227b9d1ffd3090ccb934aa89543f0bbe9f7e5b1f7a3e320c
