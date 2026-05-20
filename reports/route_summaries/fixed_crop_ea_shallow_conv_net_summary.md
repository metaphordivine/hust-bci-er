# fixed_crop_ea_shallow_conv_net Summary

route_id: fixed_crop_ea_shallow_conv_net
route_status: IDEA
audit_decision: PENDING_REMOTE_P1
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: TBD
decision: HOLD
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_shallow_conv_net.yaml --run outputs/protocol_p1_full_5lane_20260518_1225/fixed_crop_ea_shallow_conv_net --gate candidate
dataset: train_v1
split: p1 repeated group-kfold, 5 seeds x 5 folds
seed: 42, 123, 456, 789, 1024
protocol: p1_repeated_group_kfold
completed_jobs: pending remote protocol run
aggregate_metric_mean: TBD
aggregate_metric_min: TBD
aggregate_metric_max: TBD
risk notes: Placeholder summary for the remote formal P1 run so candidate audit does not fail solely on missing documentation. Replace with aggregate metrics after all 25 jobs complete; route status remains IDEA until reviewed with audit evidence.
