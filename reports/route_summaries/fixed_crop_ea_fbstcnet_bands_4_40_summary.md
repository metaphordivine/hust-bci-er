# fixed_crop_ea_fbstcnet_bands_4_40 Summary

route_id: fixed_crop_ea_fbstcnet_bands_4_40
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6944506666666667
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_fbstcnet_bands_4_40.yaml --run outputs/fixed_crop_ea_fbstcnet_bands_4_40/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: generated draft; review before committing.
audit_report_path: reports/route_summaries/evidence/fixed_crop_ea_fbstcnet_bands_4_40_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_ea_fbstcnet_bands_4_40_manifest.json
manifest_sha256: 755a4cc91adc259e43c1a02d810562876e9311f6dace5b1c7723306652dbd76d
