# Candidate Score Deep Dive

This report is diagnostic only. It does not change route status and it does not replace candidate or promoted gates.

## Inputs

| route | score_matrix | manifest |
|---|---|---|
| `fixed_crop_ea_car_fbstcnet` | `outputs/fixed_crop_ea_car_fbstcnet/candidate_20260516_seed42/score_matrix.csv` | `outputs/fixed_crop_ea_car_fbstcnet/candidate_20260516_seed42/manifest.json` |

## Leaderboard

| route | exact_single_crop_expected_BA | mean_score_top4_BA | DEP exact BA | HC exact BA | groups | trials |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_car_fbstcnet` | 0.686711 | 0.729167 | 0.522620 | 0.768756 | 12 | 96 |

## Crop Performance

| route | crop_0 | crop_1 | crop_2 | crop_3 | crop_4 | best | worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_car_fbstcnet` | 0.645833 | 0.645833 | 0.729167 | 0.687500 | 0.708333 | crop_2 | crop_0 |

## Calibration

| route | mean | std | min | max | abs_from_0.5 | overconfident_frac | pos_mean | neg_mean | brier | outside_0_1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_car_fbstcnet` | 0.510137 | 0.187145 | 0.096637 | 0.920119 | 0.150650 | 0.000000 | 0.588338 | 0.431935 | 0.206925 | 0.000000 |

## Score Correlation

| route | `fixed_crop_ea_car_fbstcnet` |
|---|---:|
| `fixed_crop_ea_car_fbstcnet` | 1.000000 |

## Mean-score Top-4 Disagreement

| route | `fixed_crop_ea_car_fbstcnet` |
|---|---:|
| `fixed_crop_ea_car_fbstcnet` | 0.000000 |

## Frequent Error Samples

| subject | trial | y_true | wrong_models | total_models | models |
|---|---|---:|---:|---:|---|
| `DEP1018` | `DEP1018_neu2` | 0 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `DEP1018` | `DEP1018_neu3` | 0 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `DEP1018` | `DEP1018_neu4` | 0 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `DEP1018` | `DEP1018_pos1` | 1 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `DEP1018` | `DEP1018_pos3` | 1 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `DEP1018` | `DEP1018_pos4` | 1 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `DEP1023` | `DEP1023_neu3` | 0 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `DEP1023` | `DEP1023_neu4` | 0 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `DEP1023` | `DEP1023_pos1` | 1 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `DEP1023` | `DEP1023_pos2` | 1 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `DEP1025` | `DEP1025_neu1` | 0 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `DEP1025` | `DEP1025_pos3` | 1 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `DEP1034` | `DEP1034_neu1` | 0 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `DEP1034` | `DEP1034_neu2` | 0 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `DEP1034` | `DEP1034_pos3` | 1 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `DEP1034` | `DEP1034_pos4` | 1 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `HC1005` | `HC1005_neu2` | 0 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `HC1005` | `HC1005_pos3` | 1 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `HC1011` | `HC1011_neu3` | 0 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `HC1011` | `HC1011_pos4` | 1 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `HC1013` | `HC1013_neu3` | 0 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `HC1013` | `HC1013_pos3` | 1 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `HC1015` | `HC1015_neu2` | 0 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `HC1015` | `HC1015_pos4` | 1 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `HC1047` | `HC1047_neu3` | 0 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
| `HC1047` | `HC1047_pos1` | 1 | 1 | 1 | `fixed_crop_ea_car_fbstcnet` |
