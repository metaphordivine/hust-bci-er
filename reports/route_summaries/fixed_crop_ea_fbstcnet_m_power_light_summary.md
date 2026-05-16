# fixed_crop_ea_fbstcnet_m_power_light Summary

route_id: fixed_crop_ea_fbstcnet_m_power_light
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6872013333333334
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_fbstcnet_m_power_light.yaml --run outputs/fixed_crop_ea_fbstcnet_m_power_light/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: generated draft; review before committing.
audit_report_path: reports/route_summaries/evidence/fixed_crop_ea_fbstcnet_m_power_light_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_ea_fbstcnet_m_power_light_manifest.json
manifest_sha256: a1d1b5d780fb0fc83ef5b902f173394a7acad2654d6371d94bd6b91886f0730e
