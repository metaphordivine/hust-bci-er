# sliding_ea_dual_graph_conformer_w6_s1 Summary

route_id: sliding_ea_dual_graph_conformer_w6_s1
route_status: IDEA
audit_decision: P3_outer0_candidate_PASS; protocol_roots_DIAGNOSTIC_ONLY
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.65088016
decision: HOLD
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/sliding_ea_dual_graph_conformer_w6_s1.yaml --run outputs/remote_C_dual_graph_conformer_p3_20260521_1727/job_runs/p3__sliding_ea_dual_graph_conformer_w6_s1__outer0__final --gate candidate
candidate_gate_sample: PASS for P3 outer0 final job after adding the dual_graph_conformer implementation evidence mapping
dataset: train_v1
split: P1 repeated group-kfold 3 seeds x 5 folds; P2 crop stress; P3 nested selection
seed: P1 seeds 42, 2026, 3407
protocol: p1_repeated_group_kfold; p2_pseudo_public_holdout; p3_nested_selection
completed_jobs: P1 15/15; P2 7/7 prediction jobs plus 1 train artifact job; P3 5/5 final jobs plus 15 inner selection jobs
P1_3x5_metric_mean: 0.642325848888889
P1_3x5_metric_std_pop: 0.053238841726618934
P1_3x5_metric_min: 0.5505933333333333
P1_3x5_metric_max: 0.7254466666666667
P2_metric_mean: 0.6339285714285714
P2_metric_std_pop: 0.05206207048969017
P2_metric_min: 0.5208333333333334
P2_metric_max: 0.6875
P3_final_metric_mean: 0.65088016
P3_final_metric_std_pop: 0.03271671662248113
P3_final_metric_min: 0.5989866666666668
P3_final_metric_max: 0.7004600000000001
risk notes: P1/P2/P3 protocol summaries are complete with no missing artifacts reported by summarize_protocol_run, and a representative P3 final job passes the candidate audit gate after the implementation evidence mapping fix. Aggregate performance is below the stronger FBSTCNet-family routes. Literature and architecture references are design priors only and are not treated as evidence of repository validity. The route remains IDEA and HOLD; use this result as a diversity/fusion or diagnostic component unless later controlled runs show a clear robustness gain.
remote_worktree: /root/autodl-tmp/hust-bci-er-dual-graph-conformer-20260521-1641
remote_run_root_P1_3x5: /root/autodl-tmp/hust-bci-er-dual-graph-conformer-20260521-1641/outputs/remote_C_dual_graph_conformer_p1_3x5b_20260521_1652
remote_run_root_P2: /root/autodl-tmp/hust-bci-er-dual-graph-conformer-20260521-1641/outputs/remote_C_dual_graph_conformer_p2_20260521_1722
remote_run_root_P3: /root/autodl-tmp/hust-bci-er-dual-graph-conformer-20260521-1641/outputs/remote_C_dual_graph_conformer_p3_20260521_1727
