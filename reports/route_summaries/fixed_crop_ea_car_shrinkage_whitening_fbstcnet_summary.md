# fixed_crop_ea_car_shrinkage_whitening_fbstcnet Summary

route_id: fixed_crop_ea_car_shrinkage_whitening_fbstcnet
route_status: IDEA
audit_decision: PENDING_FULL_PROTOCOL
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: pending
decision: PENDING
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_car_shrinkage_whitening_fbstcnet.yaml --run outputs/remote_fbstcnet_shrinkage_full_20260520_134738/p1/fixed_crop_ea_car_shrinkage_whitening_fbstcnet/job_runs/p1__fixed_crop_ea_car_shrinkage_whitening_fbstcnet__seed42__fold0 --gate candidate
dataset: train_v1
split: p1_seed42_fold0 protocol materialization
seed: [42, 123, 456, 789, 1024]
protocol: p1_repeated_group_kfold, then p2_pseudo_public_holdout and p3_nested_selection if P1 audit passes
risk notes: This is a fixed-crop FBSTCNet-M route with preprocessing ordered as EA, CAR, then shrinkage whitening. It tests whether shrinkage-stabilized whitening improves the CAR/EA FBSTCNet family without changing split definitions or evaluation protocol. The remote P1 run has produced candidate artifacts, but full P1/P2/P3 evidence and aggregate metrics are still pending; do not promote route status from this summary alone.
