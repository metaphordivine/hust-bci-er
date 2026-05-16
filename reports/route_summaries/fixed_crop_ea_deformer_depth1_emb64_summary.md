# fixed_crop_ea_deformer_depth1_emb64 Summary

route_id: fixed_crop_ea_deformer_depth1_emb64
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6727
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_deformer_depth1_emb64.yaml --run outputs/fixed_crop_ea_deformer_depth1_emb64/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: single-split candidate only; below the FBSTCNet champion and no P1/P2/P3 repeated protocol was run; retain as architecture-diversity/control evidence pending multi-seed confirmation.
audit_report_path: reports/route_summaries/evidence/fixed_crop_ea_deformer_depth1_emb64_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_ea_deformer_depth1_emb64_manifest.json
manifest_sha256: d9892d4d160fab048e9e12f4b84068167d7fc1bd58f9893f18e69654a1e82ab3
