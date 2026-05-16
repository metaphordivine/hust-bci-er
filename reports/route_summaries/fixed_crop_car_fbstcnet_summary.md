# fixed_crop_car_fbstcnet Summary

route_id: fixed_crop_car_fbstcnet
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.7064570666666666
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_car_fbstcnet.yaml --run outputs/fixed_crop_car_fbstcnet/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: generated draft; review before committing.
audit_report_path: outputs/fixed_crop_car_fbstcnet/candidate_20260516_seed42/audit_report.json
manifest_path: outputs/fixed_crop_car_fbstcnet/candidate_20260516_seed42/manifest.json
manifest_sha256: 47859b80e10520cb62c1b1de00d7264c3e942d4d830778a615899430794734bd
