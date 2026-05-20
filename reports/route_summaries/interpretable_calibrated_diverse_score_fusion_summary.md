# interpretable_calibrated_diverse_score_fusion Summary

route_id: interpretable_calibrated_diverse_score_fusion
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6977498346666666
decision: HOLD
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/interpretable_calibrated_diverse_score_fusion.yaml --run outputs/remote_interpretable_calibrated_fusion_20260520_1910/interpretable_calibrated_diverse_score_fusion --gate candidate
dataset: train_v1
split: p1 repeated group-kfold, 5 seeds x 5 folds merged from component source manifests
source_split: merged_source_protocol_splits
seed: 42, 123, 456, 789, 1024
protocol: p1_repeated_group_kfold
source_components: fixed_crop_ea_fbstcnet_component, srfnet_long_component, conformer_component, dgcnn_dann_cohort_component, dgcnn_coral_cohort_component, riemannian_tangent_component
source_manifest_count: 150
unique_source_split_count: 25
score_matrix_evidence: genuine
risk notes: P1 score-fusion candidate assembled from genuine 25-job component score matrices for all six components. The first remote audit pass was blocked only because this route summary was missing; after adding the summary, candidate audit passed. Route status remains IDEA because P2/P3 stress and nested weight confirmation have not been run.
remote_worktree: /root/autodl-tmp/hust-bci-er-interpretable-calibrated-fusion-20260520-1910
remote_run_root: outputs/remote_interpretable_calibrated_fusion_20260520_1910/interpretable_calibrated_diverse_score_fusion
