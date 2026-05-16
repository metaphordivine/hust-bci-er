# fixed_crop_whitening_eps1e3_fbstcnet Summary

route_id: fixed_crop_whitening_eps1e3_fbstcnet
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6945493333333332
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_whitening_eps1e3_fbstcnet.yaml --run outputs/fixed_crop_whitening_eps1e3_fbstcnet/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: generated draft; review before committing.
audit_report_path: reports/route_summaries/evidence/fixed_crop_whitening_eps1e3_fbstcnet_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_whitening_eps1e3_fbstcnet_manifest.json
manifest_sha256: f6b13073419545fde69392d863dde2f49096f3e215ddb432ece46f5300dc353d
