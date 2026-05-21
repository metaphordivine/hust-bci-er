# fixed_crop_ea_fbstcnet_riem_guided_gate Summary

route_id: fixed_crop_ea_fbstcnet_riem_guided_gate
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6883451306666666
decision: HOLD
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_fbstcnet_riem_guided_gate.yaml --run outputs/remote_fbstcnet_riem_gate_p1_full_20260520_2000/fixed_crop_ea_fbstcnet_riem_guided_gate/job_runs/p1__fixed_crop_ea_fbstcnet_riem_guided_gate__seed42__fold0 --gate candidate
dataset: train_v1
split: p1 repeated group-kfold, 5 seeds x 5 folds
seed: 42, 123, 456, 789, 1024
protocol: p1_repeated_group_kfold; p2_pseudo_public_holdout; p3_nested_selection
completed_jobs: P1 25/25; P2 7/7 prediction jobs plus 1 train artifact job; P3 5/5 final jobs plus 15 inner selection jobs
aggregate_metric_mean: 0.6883451306666667
aggregate_metric_std_pop: 0.04606313161271889
aggregate_metric_min: 0.591428
aggregate_metric_max: 0.7764253333333334
P1_full_metric_mean: 0.6883451306666666
P1_full_metric_std_pop: 0.04606313161271889
P1_full_metric_min: 0.591428
P1_full_metric_max: 0.7764253333333334
P1_3x5_backfill_metric_mean: 0.69485728
P1_3x5_backfill_metric_std_pop: 0.04246245267472554
P1_3x5_backfill_metric_min: 0.618412
P1_3x5_backfill_metric_max: 0.7764253333333334
P2_metric_mean: 0.6785714285714286
P2_metric_std_pop: 0.12542444943872308
P2_metric_min: 0.375
P2_metric_max: 0.75
P3_final_metric_mean: 0.7042741333333333
P3_final_metric_std_pop: 0.029800042399373283
P3_final_metric_min: 0.6727143999999999
P3_final_metric_max: 0.7576285333333334
risk notes: P1 full artifacts completed for all 25 jobs and the P1 3x5 backfill is consistent. P2 and P3 candidate protocol summaries are complete with no missing artifacts or fail markers. P2 stress remains unstable, with the weakest crop policy at 0.375 and high protocol std, while P3 nested-selection final jobs are competitive. Route status remains IDEA and decision remains HOLD because the Riemannian gate improves nested-selection viability but has not resolved crop-policy stress robustness.
remote_worktree: /root/autodl-tmp/hust-bci-er-fbstcnet-riem-gate-20260520-1958
remote_run_root_P1_full: /root/autodl-tmp/hust-bci-er-fbstcnet-riem-gate-20260520-1958/outputs/remote_fbstcnet_riem_gate_p1_full_20260520_2000/fixed_crop_ea_fbstcnet_riem_guided_gate
remote_run_root_P1_3x5_backfill: /root/autodl-tmp/hust-bci-er-codex-fbstcnet-riem-gate/outputs/remote_A_p1_3x5_throttled_extra40_20260521_0931/fixed_crop_ea_fbstcnet_riem_guided_gate
remote_run_root_P2: /root/autodl-tmp/hust-bci-er-codex-fbstcnet-riem-gate/outputs/remote_A_p2_priority_extra26_20260521_0745/fixed_crop_ea_fbstcnet_riem_guided_gate
remote_run_root_P3: /root/autodl-tmp/hust-bci-er-codex-fbstcnet-riem-gate/outputs/remote_A_p3_top_p2_extra34_20260521_0807/fixed_crop_ea_fbstcnet_riem_guided_gate
