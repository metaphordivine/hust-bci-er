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
risk notes: single-split candidate champion; no P1/P2/P3 repeated protocol; DEP exact BA 0.560967 trails HC exact BA 0.779202, so promotion needs cohort-gap and multi-seed/group-kfold confirmation.
audit_report_path: reports/route_summaries/evidence/fixed_crop_car_fbstcnet_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_car_fbstcnet_manifest.json
manifest_sha256: 29a7762011a9c9925eba209f708b07a2ea51345d016619598fcf99e03ead0f15
