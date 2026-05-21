# dgcnn_adaptation_fbstcnet_srfnet_conformer_score_average Summary

route_id: dgcnn_adaptation_fbstcnet_srfnet_conformer_score_average
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.6984882666666666
decision: PASS for seed42-aligned score-fusion evidence; keep route status unchanged until a broader audit-backed promotion decision exists.
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/dgcnn_adaptation_fbstcnet_srfnet_conformer_score_average.yaml --run outputs/remote_adaptation_score_fusion_full_20260520_161029/score_fusion_seed42_aligned/dgcnn_adaptation_fbstcnet_srfnet_conformer_score_average --gate candidate
dataset: train_v1
split: p1_seed42_fold0__score_fusion_sources
source_split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold seed42/fold0-4 aligned subset
source_components: dgcnn_dann_cohort_component, dgcnn_coral_cohort_component, fixed_crop_ea_fbstcnet_component, srfnet_long_component, conformer_component
risk notes: DANN/CORAL reusable component evidence currently exists only for seed42/fold0-4. The first full 25-job assembly correctly failed key alignment against the multi-seed FBSTCNet/SRFNet/Conformer component tables, so this summary records the auditable seed42-aligned recovery run rather than pretending complete five-seed component coverage exists.
remote_run_root: /root/autodl-tmp/hust-bci-er-route-adaptation-score-fusion-20260520-1610/outputs/remote_adaptation_score_fusion_full_20260520_161029
remote_score_fusion_run: outputs/remote_adaptation_score_fusion_full_20260520_161029/score_fusion_seed42_aligned/dgcnn_adaptation_fbstcnet_srfnet_conformer_score_average
