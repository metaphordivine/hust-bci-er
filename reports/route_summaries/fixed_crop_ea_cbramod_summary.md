# fixed_crop_ea_cbramod Summary

route_id: fixed_crop_ea_cbramod
route_status: IDEA
audit_decision: PENDING_REAUDIT
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.641108085333333
decision: HOLD
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_cbramod.yaml --run outputs/protocol_p1_full_all_retry/fixed_crop_ea_cbramod/job_runs/p1__fixed_crop_ea_cbramod__seed42__fold0 --gate candidate
dataset: train_v1
split: p1 repeated group-kfold, 5 seeds x 5 folds
seed: 42, 123, 456, 789, 1024
protocol: p1_repeated_group_kfold
completed_jobs: 25
aggregate_metric_mean: 0.641108085333333
aggregate_metric_min: 0.567650133333333
aggregate_metric_max: 0.7089
risk notes: P1 full retry artifacts completed for all 25 jobs, and the only observed candidate audit blocker was the missing route summary. Rerun the candidate audit before treating this as PASS; route status remains IDEA and this is not promotion evidence.
