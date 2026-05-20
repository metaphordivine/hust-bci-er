# fixed_crop_ea_fbstcnet_riem_guided_gate Summary

route_id: fixed_crop_ea_fbstcnet_riem_guided_gate
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6883451306666667
decision: HOLD
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_fbstcnet_riem_guided_gate.yaml --run outputs/remote_fbstcnet_riem_gate_p1_full_20260520_2000/fixed_crop_ea_fbstcnet_riem_guided_gate/job_runs/p1__fixed_crop_ea_fbstcnet_riem_guided_gate__seed42__fold0 --gate candidate
dataset: train_v1
split: p1 repeated group-kfold, 5 seeds x 5 folds
seed: 42, 123, 456, 789, 1024
protocol: p1_repeated_group_kfold
completed_jobs: 25
aggregate_metric_mean: 0.6883451306666667
aggregate_metric_std_pop: 0.04606313161271889
aggregate_metric_min: 0.591428
aggregate_metric_max: 0.7764253333333334
risk notes: P1 full artifacts completed for all 25 jobs. The first remote audit pass was blocked only because this route summary was missing; after adding the summary, rerunning the protocol command skipped all existing artifacts and returned failed_jobs=0. Route status remains IDEA because P2/P3 stress and nested-selection evidence have not been run.
remote_worktree: /root/autodl-tmp/hust-bci-er-fbstcnet-riem-gate-20260520-1958
remote_run_root: outputs/remote_fbstcnet_riem_gate_p1_full_20260520_2000/fixed_crop_ea_fbstcnet_riem_guided_gate
