# fixed_crop_ea_fbstcnet_m_conn_light Summary

route_id: fixed_crop_ea_fbstcnet_m_conn_light
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6876333333333333
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_fbstcnet_m_conn_light.yaml --run outputs/fixed_crop_ea_fbstcnet_m_conn_light/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: single-split FBSTCNet ablation evidence only; no P1/P2/P3 repeated protocol was run, and promotion requires multi-seed/group-kfold, DEP/HC gap, and hard-subject review.
audit_report_path: reports/route_summaries/evidence/fixed_crop_ea_fbstcnet_m_conn_light_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_ea_fbstcnet_m_conn_light_manifest.json
manifest_sha256: c4c62eb2de0c2fb41a288e6f38fe06f43b2093528f585c0622a4e7482d4a8eaa
