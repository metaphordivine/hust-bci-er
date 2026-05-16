# srfnet_as_query_fbstcnet_context_fusion Summary

route_id: srfnet_as_query_fbstcnet_context_fusion
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6707386666666668
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/srfnet_as_query_fbstcnet_context_fusion.yaml --run outputs/score_fusion_fbstcnet/20260516_fixed_ea_component/srfnet_as_query_fbstcnet_context_fusion --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: single-split score-fusion candidate only; fusion score is a ranking score, not a calibrated probability, and components use fixed_crop_ea_fbstcnet rather than the new CAR/C-only champion; rerun with new champion components before promotion.
audit_report_path: reports/route_summaries/evidence/srfnet_as_query_fbstcnet_context_fusion_audit_report.json
manifest_path: reports/route_summaries/evidence/srfnet_as_query_fbstcnet_context_fusion_manifest.json
manifest_sha256: 0702ead12b3540e8187a21739bec34c2c76de6ecfb7568fc76b8f78349af2ea7
