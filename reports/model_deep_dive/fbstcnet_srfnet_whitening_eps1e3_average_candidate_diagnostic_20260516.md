# Candidate Score Deep Dive

This report is diagnostic only. It does not change route status and it does not replace candidate or promoted gates.

## Inputs

| route | score_matrix | manifest |
|---|---|---|
| `fbstcnet_srfnet_whitening_eps1e3_average` | `outputs/score_fusion_fbstcnet/20260516_fixed_ea_component/fbstcnet_srfnet_whitening_eps1e3_average/score_matrix.csv` | `outputs/score_fusion_fbstcnet/20260516_fixed_ea_component/fbstcnet_srfnet_whitening_eps1e3_average/manifest.json` |

## Leaderboard

| route | exact_single_crop_expected_BA | mean_score_top4_BA | DEP exact BA | HC exact BA | groups | trials |
|---|---:|---:|---:|---:|---:|---:|
| `fbstcnet_srfnet_whitening_eps1e3_average` | 0.699401 | 0.729167 | 0.473244 | 0.812480 | 12 | 96 |

## Crop Performance

| route | crop_0 | crop_1 | crop_2 | crop_3 | crop_4 | best | worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fbstcnet_srfnet_whitening_eps1e3_average` | 0.687500 | 0.687500 | 0.708333 | 0.687500 | 0.687500 | crop_2 | crop_0 |

## Calibration

| route | mean | std | min | max | abs_from_0.5 | overconfident_frac | pos_mean | neg_mean | brier | outside_0_1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fbstcnet_srfnet_whitening_eps1e3_average` | 0.000000 | 0.790858 | -1.298763 | 1.720953 | 0.808130 |  | 0.351567 | -0.351567 |  | 0.645833 |

## Score Correlation

| route | `fbstcnet_srfnet_whitening_eps1e3_average` |
|---|---:|
| `fbstcnet_srfnet_whitening_eps1e3_average` | 1.000000 |

## Mean-score Top-4 Disagreement

| route | `fbstcnet_srfnet_whitening_eps1e3_average` |
|---|---:|
| `fbstcnet_srfnet_whitening_eps1e3_average` | 0.000000 |

## Frequent Error Samples

| subject | trial | y_true | wrong_models | total_models | models |
|---|---|---:|---:|---:|---|
| `DEP1018` | `DEP1018_neu2` | 0 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `DEP1018` | `DEP1018_neu3` | 0 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `DEP1018` | `DEP1018_neu4` | 0 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `DEP1018` | `DEP1018_pos1` | 1 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `DEP1018` | `DEP1018_pos3` | 1 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `DEP1018` | `DEP1018_pos4` | 1 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `DEP1023` | `DEP1023_neu3` | 0 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `DEP1023` | `DEP1023_neu4` | 0 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `DEP1023` | `DEP1023_pos1` | 1 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `DEP1023` | `DEP1023_pos2` | 1 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `DEP1025` | `DEP1025_neu1` | 0 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `DEP1025` | `DEP1025_pos3` | 1 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `DEP1034` | `DEP1034_neu1` | 0 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `DEP1034` | `DEP1034_neu2` | 0 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `DEP1034` | `DEP1034_neu3` | 0 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `DEP1034` | `DEP1034_pos2` | 1 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `DEP1034` | `DEP1034_pos3` | 1 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `DEP1034` | `DEP1034_pos4` | 1 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `HC1005` | `HC1005_neu2` | 0 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `HC1005` | `HC1005_pos3` | 1 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `HC1011` | `HC1011_neu3` | 0 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `HC1011` | `HC1011_pos4` | 1 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `HC1015` | `HC1015_neu2` | 0 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `HC1015` | `HC1015_pos4` | 1 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `HC1047` | `HC1047_neu3` | 0 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
| `HC1047` | `HC1047_pos2` | 1 | 1 | 1 | `fbstcnet_srfnet_whitening_eps1e3_average` |
