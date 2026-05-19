# sliding_ea_cbramod_pretrained_w3p2_s1p7 Summary

route_id: sliding_ea_cbramod_pretrained_w3p2_s1p7
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.628454
decision: HOLD_FOR_REVIEW
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/sliding_ea_cbramod_pretrained_w3p2_s1p7.yaml --run outputs/autodl_cbramod_20260519/remote_cbramod_pretrained_p1_20260519/sliding_ea_cbramod_pretrained_w3p2_s1p7 --gate candidate
dataset: train_v1
split: p1 repeated group-kfold, 5 seeds x 5 folds
seed: 42, 123, 456, 789, 1024
protocol: p1_repeated_group_kfold
risk notes: Official CBraMod pretrained finetune route completed formal P1/P2/P3 evidence on AutoDL. Route status remains IDEA pending human review; ignored checkpoint scratch/model_weights/cbramod/pretrained_weights.pth is required and audited by SHA256.
completed_jobs: 25
aggregate_metric_mean: 0.628454
aggregate_metric_min: 0.569553
aggregate_metric_max: 0.713124
p2_metric_mean: 0.583333
p3_metric_mean: 0.640797
comparison_report: reports/model_deep_dive/cbramod_pretrained_vs_scratch_p1_p2_p3_20260519.md
