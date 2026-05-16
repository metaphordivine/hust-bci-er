# fixed_crop_ea_fbstcnet Summary

route_id: fixed_crop_ea_fbstcnet
route_status: CANDIDATE
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.686144
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_fbstcnet.yaml --run outputs/model_route_exploration/20260516_candidate/fixed_crop_ea_fbstcnet --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: candidate audit passed; improves over sliding_window_fbstcnet 0.602884 by +0.083260; compact evidence is stored under reports.
audit_report_path: reports/route_summaries/evidence/fixed_crop_ea_fbstcnet_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_ea_fbstcnet_manifest.json
manifest_sha256: 0b2ab35375070515def5ff0dab8a1c2dcc8196b85dc4e555b55efe75f39c9f9b
