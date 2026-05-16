# fixed_crop_ea_deformer_depth2_emb96 Summary

route_id: fixed_crop_ea_deformer_depth2_emb96
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6563674666666668
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_deformer_depth2_emb96.yaml --run outputs/fixed_crop_ea_deformer_depth2_emb96/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: generated draft; review before committing.
audit_report_path: reports/route_summaries/evidence/fixed_crop_ea_deformer_depth2_emb96_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_ea_deformer_depth2_emb96_manifest.json
manifest_sha256: ebfd386dc3980824ad96e6ca30908871bf40499442bc5482c739bf9b0bf36e8e
