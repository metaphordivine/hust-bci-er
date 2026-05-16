# fbstcnet_srfnet_whitening_eps1e3_average Summary

route_id: fbstcnet_srfnet_whitening_eps1e3_average
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6994013333333333
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fbstcnet_srfnet_whitening_eps1e3_average.yaml --run outputs/score_fusion_fbstcnet/20260516_fixed_ea_component/fbstcnet_srfnet_whitening_eps1e3_average --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: single-split score-fusion candidate only; fusion score is a ranking score, not a calibrated probability, and components use fixed_crop_ea_fbstcnet rather than the new CAR/C-only champion; rerun with new champion components before promotion.
audit_report_path: reports/route_summaries/evidence/fbstcnet_srfnet_whitening_eps1e3_average_audit_report.json
manifest_path: reports/route_summaries/evidence/fbstcnet_srfnet_whitening_eps1e3_average_manifest.json
manifest_sha256: 72c77e06dff84977590b5c3dc6d4a973e31023001eece182683c5b0b964bc347
