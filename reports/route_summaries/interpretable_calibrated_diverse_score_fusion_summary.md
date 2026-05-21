# interpretable_calibrated_diverse_score_fusion Summary

route_id: interpretable_calibrated_diverse_score_fusion
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value_p1: 0.6977498346666666
primary_metric_value_p3_final: 0.7026709333333334
decision: HOLD
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/interpretable_calibrated_diverse_score_fusion.yaml --run outputs/remote_interpretable_calibrated_fusion_20260520_1910/interpretable_calibrated_diverse_score_fusion --gate candidate
dataset: train_v1
split: P1 repeated group-kfold, P2 crop stress holdout, and P3 outer-final nested selection evidence assembled from component source manifests
source_split: merged_source_protocol_splits for P1/P3; copied holdout source split for each P2 crop policy
seed: P1 42/123/456/789/1024; P2 train seed 42; P3 outer-final seeds 42-46
protocol: p1_repeated_group_kfold + p2_pseudo_public_holdout + p3_nested_selection
source_components: fixed_crop_ea_fbstcnet_component, srfnet_long_component, conformer_component, dgcnn_dann_cohort_component, dgcnn_coral_cohort_component, riemannian_tangent_component
source_manifest_count: P1 150; P2 6 per crop policy; P3 30
unique_source_split_count: P1 25; P2 1 per crop policy; P3 5
score_matrix_evidence: genuine
risk notes: P1 score-fusion candidate assembled from genuine 25-job component score matrices for all six components. P2 and P3 were assembled after preserving selected-crop provenance in score-fusion component exports; all candidate audits now pass. Route status remains IDEA because route-status promotion should be handled by a separate audit/status decision.
remote_worktree: /root/autodl-tmp/hust-bci-er-interpretable-calibrated-fusion-20260520-1910
remote_run_root: outputs/remote_interpretable_calibrated_fusion_20260520_1910/interpretable_calibrated_diverse_score_fusion
remote_p2_p3_assembly_root: /root/autodl-tmp/hust-bci-er-pr30-assembly-20260521-1405/outputs/pr30_fusion_assembly_20260521_1432

## P2 Crop Stress

| policy | candidate audit | exact_single_crop_expected_BA |
|---|---:|---:|
| crop1 | PASS | 0.7500000000000000 |
| crop2 | PASS | 0.7708333333333334 |
| crop3 | PASS | 0.7291666666666666 |
| crop4 | PASS | 0.7291666666666666 |
| crop5 | PASS | 0.7291666666666666 |
| random | PASS | 0.7083333333333334 |
| worst | PASS | 0.5625000000000000 |

## P3 Nested Selection

candidate audit: PASS
outer-final exact_single_crop_expected_BA: 0.7026709333333334
source manifests: 30 genuine component score-matrix manifests across 5 outer-final splits.
