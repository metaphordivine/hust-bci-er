# fbstcnet_srfnet_score_average Summary

route_id: fbstcnet_srfnet_score_average
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6931226666666667
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fbstcnet_srfnet_score_average.yaml --run outputs/score_fusion_fbstcnet/20260516_fixed_ea_component/fbstcnet_srfnet_score_average --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: generated draft; review before committing.
audit_report_path: reports/route_summaries/evidence/fbstcnet_srfnet_score_average_audit_report.json
manifest_path: reports/route_summaries/evidence/fbstcnet_srfnet_score_average_manifest.json
manifest_sha256: 5b04482e5690198b4e647e5b5818d8d4f2e7bfe263b405f3b58008a0ad5d0147
