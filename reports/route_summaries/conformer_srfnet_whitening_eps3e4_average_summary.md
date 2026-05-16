# conformer_srfnet_whitening_eps3e4_average Summary

route_id: conformer_srfnet_whitening_eps3e4_average
route_status: CANDIDATE
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6139133333333333
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/conformer_srfnet_whitening_eps3e4_average.yaml --run outputs/score_fusion_full/20260516_135417/conformer_srfnet_whitening_eps3e4_average --gate candidate
dataset: train_v1
split: route_component_split_v1
seed: 42
protocol: p1_repeated_group_kfold
risk notes: score-fusion candidate assembled from PASS component score matrices; handcrafted component routes remain IDEA because no real handcrafted score artifact exists.
audit_report_path: reports/route_summaries/evidence/conformer_srfnet_whitening_eps3e4_average_audit_report.json
manifest_path: reports/route_summaries/evidence/conformer_srfnet_whitening_eps3e4_average_manifest.json
manifest_sha256: bc3edb51dddd52b205722ea479ac8898660c9435eff5e4367d822d66d387d31c