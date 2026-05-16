# fixed_crop_ea_fbstcnet_f1_24_f2_48 Summary

route_id: fixed_crop_ea_fbstcnet_f1_24_f2_48
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6649416
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_fbstcnet_f1_24_f2_48.yaml --run outputs/fixed_crop_ea_fbstcnet_f1_24_f2_48/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: generated draft; review before committing.
audit_report_path: reports/route_summaries/evidence/fixed_crop_ea_fbstcnet_f1_24_f2_48_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_ea_fbstcnet_f1_24_f2_48_manifest.json
manifest_sha256: a123e39cb3dda4fecba3cecea756c7046fb066c18be219eaa6482ac916567b79
