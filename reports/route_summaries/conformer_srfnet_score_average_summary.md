# conformer_srfnet_score_average Summary

route_id: conformer_srfnet_score_average
route_status: CANDIDATE
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.62744
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/conformer_srfnet_score_average.yaml --run outputs/score_fusion_full/20260516_135417/conformer_srfnet_score_average --gate candidate
dataset: train_v1
split: p1_seed42_fold0
source_split: route_component_split_v1
seed: 42
protocol: p1_repeated_group_kfold
risk notes: score-fusion candidate assembled from PASS component score matrices with committed compact component evidence; handcrafted component routes remain IDEA because no real handcrafted score artifact exists.
audit_report_path: reports/route_summaries/evidence/conformer_srfnet_score_average_audit_report.json
manifest_path: reports/route_summaries/evidence/conformer_srfnet_score_average_manifest.json
manifest_sha256: 0e6effd4b8b9b8737a85a0411fbee8173d670a325ffec63bb9debf72eb62ced2