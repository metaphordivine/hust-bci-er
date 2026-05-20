# FBSTCNet Shrinkage Whitening P1/P2/P3 Report

route_id: `fixed_crop_ea_car_shrinkage_whitening_fbstcnet`

PR: `#27`

remote_worktree: `/root/autodl-tmp/hust-bci-er-route-fbstcnet-shrinkage-20260520-1346`

remote_run_root: `outputs/remote_fbstcnet_shrinkage_full_20260520_134738`

git_commit: `67b98c7184b4c0ffaf4f1d5e9b057febee49006c`

## Protocol Results

| protocol | completed evidence | mean | std | min | max | audit |
|---|---:|---:|---:|---:|---:|---|
| P1 repeated group k-fold | 25/25 prediction jobs | 0.710558 | 0.043386 | 0.619360 | 0.814976 | PASS |
| P2 pseudo-public holdout | 7/7 prediction jobs, 1/1 artifact job | 0.651786 | 0.175973 | 0.229167 | 0.770833 | PASS |
| P3 nested selection | 5/5 final jobs, 15/15 selection jobs | 0.719944 | 0.026641 | 0.696168 | 0.767203 | PASS |

## Interpretation

This route completed the full P1/P2/P3 evidence loop after adding the missing route summary required by candidate audit. P1 and P3 are both strong for the FBSTCNet family, and P3 has lower spread than P1. The weak point is P2: the mean is acceptable, but the min of 0.229167 shows a severe crop-policy tail, so this should not be promoted only from the aggregate mean.

The route keeps the reviewed preprocessing order: `euclidean_alignment`, `car`, then `shrinkage_whitening`. No split definitions, labels, leaderboard feedback, or route status were changed.

## Next Use

Use this route as a strong single-model component for score-fusion exploration, but pair it with crop-policy robustness checks. In particular, route #2 should test whether fusion can preserve the P1/P3 gains while reducing the P2 worst-crop failure mode.
