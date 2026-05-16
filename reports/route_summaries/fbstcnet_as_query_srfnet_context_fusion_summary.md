# fbstcnet_as_query_srfnet_context_fusion Summary

route_id: fbstcnet_as_query_srfnet_context_fusion
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6933813333333333
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fbstcnet_as_query_srfnet_context_fusion.yaml --run outputs/score_fusion_fbstcnet/20260516_fixed_ea_component/fbstcnet_as_query_srfnet_context_fusion --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: single-split score-fusion candidate only; fusion score is a ranking score, not a calibrated probability, and components use fixed_crop_ea_fbstcnet rather than the new CAR/C-only champion; rerun with new champion components before promotion.
audit_report_path: reports/route_summaries/evidence/fbstcnet_as_query_srfnet_context_fusion_audit_report.json
manifest_path: reports/route_summaries/evidence/fbstcnet_as_query_srfnet_context_fusion_manifest.json
manifest_sha256: d5fc1b794dacd1dfc9baff3ad7404818911f25e027dbe7253dd85fb378a1db53
