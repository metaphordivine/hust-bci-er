# car_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion Summary

route_id: car_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion
route_status: IDEA
audit_decision: PASS_P1_P2_FIXED_CROPS_AND_P3
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.7307332266666667
decision: PROMOTE_TO_FULL_EXPERIMENT_TRACKING
reproduce: python scripts/run_score_fusion_routes.py --route-filter car_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion --base-runs-dir outputs/pr41_score_fusion_full_p1_20260522_1028/base_runs_p1_full --export-missing --output-dir outputs/pr41_score_fusion_full_p1_20260522_1028/p1_full --audit
dataset: train_v1
split: P1/P2 protocol runner component splits
seed: protocol default unless specified by component manifests
protocol: score_fusion query_context
risk notes: Uses genuine score_matrix components only. P1 full 3x5 and P3 assembled cleanly; the mixed P2 aggregate is diagnostic, while per-fixed-crop P2 assemblies pass candidate audit. Do not tune weights from outer results.

evidence:
  - run: P1 full 3x5
    output_root: /root/autodl-tmp/hust-bci-er-pr41-exec-B-20260522_1009/outputs/pr41_score_fusion_full_p1_20260522_1028/p1_full/car_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion
    audit: PASS
    exact_single_crop_expected_BA: 0.7307332266666667
    dep_ba: 0.64664768
    hc_ba: 0.7727759999999999
    dep_hc_ratio: 0.836785407414309
  - run: P1 seed42 screen
    output_root: /root/autodl-tmp/hust-bci-er-pr41-exec-B-20260522_1009/outputs/pr41_score_fusion_screen_20260522_101317/p1_seed42/car_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion
    audit: PASS
    exact_single_crop_expected_BA: 0.7324442666666666
    dep_hc_ratio: 0.8518715411531392
  - run: P2 diagnostic screen
    output_root: /root/autodl-tmp/hust-bci-er-pr41-exec-B-20260522_1009/outputs/pr41_score_fusion_screen_20260522_101317/p2/car_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion
    audit: DIAGNOSTIC_ONLY
    exact_single_crop_expected_BA: 0.7180239139784946
    dep_hc_ratio: 0.8658339627823937
    p2_worst_ba: 0.5
  - run: P2 fixed-crop candidate repair
    output_root: outputs/remote_fix_pr41_p2_crop_candidate_20260522_233648/p2__eval_crop*/car_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion
    audit: PASS for crop1 through crop5
    crop_exact_ba: [0.7083333, 0.7083333, 0.8125000, 0.7916667, 0.7500000]
    crop_dep_hc_ratio: [0.9565217, 0.8333333, 0.8888889, 1.0400000, 0.8800000]
  - run: P3 nested
    output_root: /root/autodl-tmp/hust-bci-er-pr41-exec-B-20260522_1009/outputs/pr41_score_fusion_full_p3_20260522_102046/p3/car_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion
    audit: PASS
    exact_single_crop_expected_BA: 0.7366978240000001
    p3_final_mean_ba: 0.73768576
    dep_hc_ratio: 0.8347032297442678
