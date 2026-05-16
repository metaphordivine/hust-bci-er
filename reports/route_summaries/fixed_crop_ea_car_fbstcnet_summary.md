# fixed_crop_ea_car_fbstcnet Summary

route_id: fixed_crop_ea_car_fbstcnet
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6867106666666666
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_car_fbstcnet.yaml --run outputs/fixed_crop_ea_car_fbstcnet/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: generated draft; review before committing.
audit_report_path: reports/route_summaries/evidence/fixed_crop_ea_car_fbstcnet_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_ea_car_fbstcnet_manifest.json
manifest_sha256: 2cb609eb50d7c98deadb85529a1e471a7c78e06fbd1c10e03e5247452ea00fa6
