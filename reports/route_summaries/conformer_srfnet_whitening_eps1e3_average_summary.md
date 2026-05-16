# conformer_srfnet_whitening_eps1e3_average Summary

route_id: conformer_srfnet_whitening_eps1e3_average
route_status: CANDIDATE
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6439666666666667
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/conformer_srfnet_whitening_eps1e3_average.yaml --run outputs/score_fusion_full/20260516_135417/conformer_srfnet_whitening_eps1e3_average --gate candidate
dataset: train_v1
split: p1_seed42_fold0
source_split: route_component_split_v1
seed: 42
protocol: p1_repeated_group_kfold
risk notes: score-fusion candidate assembled from PASS component score matrices with committed compact component evidence; handcrafted component routes remain IDEA because no real handcrafted score artifact exists.
audit_report_path: reports/route_summaries/evidence/conformer_srfnet_whitening_eps1e3_average_audit_report.json
manifest_path: reports/route_summaries/evidence/conformer_srfnet_whitening_eps1e3_average_manifest.json
manifest_sha256: cab1c39c5d044f8e268a7879d845d34fee916470dbacef9b9bad38b5640b1076