# fixed_crop_ea_deformer_dropout035 Summary

route_id: fixed_crop_ea_deformer_dropout035
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6416928
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_deformer_dropout035.yaml --run outputs/fixed_crop_ea_deformer_dropout035/candidate_20260516_seed42 --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: generated draft; review before committing.
audit_report_path: reports/route_summaries/evidence/fixed_crop_ea_deformer_dropout035_audit_report.json
manifest_path: reports/route_summaries/evidence/fixed_crop_ea_deformer_dropout035_manifest.json
manifest_sha256: cc2eece5d577b10a88523cf99a02ed276108ddee2dd1cc06df44573b659fbdc6
