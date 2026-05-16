# fixed_crop_fbstcnet_zscore_only Summary

route_id: fixed_crop_fbstcnet_zscore_only
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6957770666666666
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_fbstcnet_zscore_only.yaml --run outputs/fixed_crop_fbstcnet_zscore_only/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: single-split FBSTCNet ablation evidence only; no P1/P2/P3 repeated protocol was run, and promotion requires multi-seed/group-kfold, DEP/HC gap, and hard-subject review.
audit_report_path: reports/route_summaries/evidence/fixed_crop_fbstcnet_zscore_only_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_fbstcnet_zscore_only_manifest.json
manifest_sha256: 656377d3fb49a14c9c10dc38ba6bdc59da11f4aae0fe4d17411d65744334d501
