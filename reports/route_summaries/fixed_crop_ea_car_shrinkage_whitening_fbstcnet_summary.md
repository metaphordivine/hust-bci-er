# fixed_crop_ea_car_shrinkage_whitening_fbstcnet Summary

route_id: fixed_crop_ea_car_shrinkage_whitening_fbstcnet
route_status: IDEA
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.7199435733333333
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/fixed_crop_ea_car_shrinkage_whitening_fbstcnet.yaml --run outputs/remote_fbstcnet_shrinkage_full_20260520_134738/p1/fixed_crop_ea_car_shrinkage_whitening_fbstcnet/job_runs/p1__fixed_crop_ea_car_shrinkage_whitening_fbstcnet__seed42__fold0 --gate candidate
dataset: train_v1
split: p1_seed42_fold0 protocol materialization
seed: [42, 123, 456, 789, 1024]
protocol: p1_repeated_group_kfold, p2_pseudo_public_holdout, p3_nested_selection
completed_jobs: P1 25/25, P2 7/7 prediction jobs plus 1 artifact job, P3 5/5 final jobs plus 15 selection jobs
p1_mean: 0.7105578666666666
p1_std: 0.04338584265228565
p1_min: 0.61936
p1_max: 0.8149762666666667
p2_mean: 0.6517857142857144
p2_std: 0.17597316070350577
p2_min: 0.22916666666666666
p2_max: 0.7708333333333334
p3_mean: 0.7199435733333333
p3_std: 0.026640750549373666
p3_min: 0.696168
p3_max: 0.7672034666666665
remote_run_root: /root/autodl-tmp/hust-bci-er-route-fbstcnet-shrinkage-20260520-1346/outputs/remote_fbstcnet_shrinkage_full_20260520_134738
risk notes: This is a fixed-crop FBSTCNet-M route with preprocessing ordered as EA, CAR, then shrinkage whitening. It tests whether shrinkage-stabilized whitening improves the CAR/EA FBSTCNet family without changing split definitions or evaluation protocol. P1 and P3 are strong, but P2 has a weak worst-case crop-policy tail; route status remains IDEA until a separate status-change audit supports promotion.
