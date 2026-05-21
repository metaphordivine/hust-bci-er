# fixed_crop_ea_riemannian_tangent Summary

route_id: fixed_crop_ea_riemannian_tangent
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.5943390506666667
decision: HOLD
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_riemannian_tangent.yaml --run outputs/remote_riemannian_tangent_p1_full_20260520_1820/fixed_crop_ea_riemannian_tangent/job_runs/p1__fixed_crop_ea_riemannian_tangent__seed42__fold0 --gate candidate
dataset: train_v1
split: p1 repeated group-kfold, 5 seeds x 5 folds
seed: 42, 123, 456, 789, 1024
protocol: p1_repeated_group_kfold
completed_jobs: 25
aggregate_metric_mean: 0.5943390506666667
aggregate_metric_std_pop: 0.03347838151830074
aggregate_metric_min: 0.5184951999999999
aggregate_metric_max: 0.6618464
risk notes: P1 full artifacts completed for all 25 jobs. The first remote audit pass was blocked only because this route summary was missing; after adding the summary, rerunning the protocol command skipped existing artifacts and returned failed_jobs=0. Route status remains IDEA and this is not promotion evidence.
remote_worktree: /root/autodl-tmp/hust-bci-er-route-riemannian-component-20260520-1820
remote_run_root: outputs/remote_riemannian_tangent_p1_full_20260520_1820/fixed_crop_ea_riemannian_tangent
