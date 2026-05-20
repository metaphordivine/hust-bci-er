# fixed_crop_ea_fbcnet Summary

route_id: fixed_crop_ea_fbcnet
route_status: IDEA
audit_decision: PENDING_RERUN
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6449383888888889
decision: HOLD
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_fbcnet.yaml --run outputs/protocol_p1_full_5lane_20260518_1225/fixed_crop_ea_fbcnet/job_runs/p1__fixed_crop_ea_fbcnet__seed42__fold0 --gate candidate
dataset: train_v1
split: p1 repeated group-kfold, 5 seeds x 5 folds
seed: 42, 123, 456, 789, 1024
protocol: p1_repeated_group_kfold
completed_jobs: 24
aggregate_metric_mean_partial: 0.6449383888888889
aggregate_metric_min_partial: 0.5707402666666667
aggregate_metric_max_partial: 0.7279458666666666
risk notes: Remote P1 run was interrupted by fail-fast after DGCNN audit blocking; this route has partial artifacts only and must finish the remaining jobs and rerun candidate audit before any decision.
