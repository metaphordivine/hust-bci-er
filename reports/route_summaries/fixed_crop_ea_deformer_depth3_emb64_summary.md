# fixed_crop_ea_deformer_depth3_emb64 Summary

route_id: fixed_crop_ea_deformer_depth3_emb64
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6457802666666667
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_deformer_depth3_emb64.yaml --run outputs/fixed_crop_ea_deformer_depth3_emb64/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: single-split candidate only; below the FBSTCNet champion and no P1/P2/P3 repeated protocol was run; retain as architecture-diversity/control evidence pending multi-seed confirmation.
audit_report_path: reports/route_summaries/evidence/fixed_crop_ea_deformer_depth3_emb64_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_ea_deformer_depth3_emb64_manifest.json
manifest_sha256: 9734b871c4d0a7200681ce564e4828d717d9bf69ae148c2320a7031cf41e6e61
