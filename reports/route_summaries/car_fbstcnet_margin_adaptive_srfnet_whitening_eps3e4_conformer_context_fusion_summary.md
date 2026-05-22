# car_fbstcnet_margin_adaptive_srfnet_whitening_eps3e4_conformer_context_fusion Summary

route_id: car_fbstcnet_margin_adaptive_srfnet_whitening_eps3e4_conformer_context_fusion
route_status: IDEA
audit_decision: PASS_P1_FULL_AND_P3_DIAGNOSTIC_P2
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.7267276088888889
decision: HOLD_FOR_DIAGNOSTIC_AND_WORST_CROP_SIGNAL
reproduce: python scripts/run_score_fusion_routes.py --route-filter car_fbstcnet_margin_adaptive_srfnet_whitening_eps3e4_conformer_context_fusion --base-runs-dir /root/autodl-tmp/hust-bci-er-pr41-exec-B-20260522_1009/outputs/pr41_score_fusion_full_p1_20260522_1028/base_runs_p1_full --export-missing --output-dir outputs/pr42_margin_adaptive_full_20260522_1118/p1_full --audit
dataset: train_v1
split: P1/P2 protocol runner component splits
seed: protocol default unless specified by component manifests
protocol: score_fusion margin_adaptive_query_context
risk notes: Uses genuine PR41 component score matrices only. The adaptive margin rule improves P2 worst-crop BA but lowers full P1/P3 mean BA and DEP/HC margin relative to PR41, so keep it as a diagnostic idea rather than a replacement route. Do not tune adaptive parameters from outer results.

evidence:
  - run: P1 full 3x5
    output_root: /root/autodl-tmp/hust-bci-er-pr42-exec-B-20260522_1114/outputs/pr42_margin_adaptive_full_20260522_1118/p1_full/car_fbstcnet_margin_adaptive_srfnet_whitening_eps3e4_conformer_context_fusion
    audit: PASS
    exact_single_crop_expected_BA: 0.7267276088888889
    dep_ba: 0.6306844266666668
    hc_ba: 0.7747491999999999
    dep_hc_ratio: 0.8140497940064564
  - run: P1 seed42 screen
    output_root: /root/autodl-tmp/hust-bci-er-pr42-exec-B-20260522_1114/outputs/pr42_margin_adaptive_screen_20260522_1115/p1_seed42/car_fbstcnet_margin_adaptive_srfnet_whitening_eps3e4_conformer_context_fusion
    audit: PASS
    exact_single_crop_expected_BA: 0.726472
    dep_hc_ratio: 0.8227866660380615
  - run: P2 diagnostic screen
    output_root: /root/autodl-tmp/hust-bci-er-pr42-exec-B-20260522_1114/outputs/pr42_margin_adaptive_screen_20260522_1115/p2/car_fbstcnet_margin_adaptive_srfnet_whitening_eps3e4_conformer_context_fusion
    audit: DIAGNOSTIC_ONLY
    exact_single_crop_expected_BA: 0.7112679569892475
    dep_hc_ratio: 0.827606583815545
    p2_worst_ba: 0.5208333333333334
  - run: P3 nested
    output_root: /root/autodl-tmp/hust-bci-er-pr42-exec-B-20260522_1114/outputs/pr42_margin_adaptive_full_20260522_1118/p3/car_fbstcnet_margin_adaptive_srfnet_whitening_eps3e4_conformer_context_fusion
    audit: PASS
    exact_single_crop_expected_BA: 0.73453392
    p3_final_mean_ba: 0.73698944
    dep_hc_ratio: 0.818592656373237
