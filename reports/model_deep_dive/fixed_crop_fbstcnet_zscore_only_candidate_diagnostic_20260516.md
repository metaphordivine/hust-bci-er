# Candidate Score Deep Dive

This report is diagnostic only. It does not change route status and it does not replace candidate or promoted gates.

## Inputs

| route | score_matrix | manifest |
|---|---|---|
| `fixed_crop_fbstcnet_zscore_only` | `outputs/fixed_crop_fbstcnet_zscore_only/candidate_20260516_seed42/score_matrix.csv` | `outputs/fixed_crop_fbstcnet_zscore_only/candidate_20260516_seed42/manifest.json` |

## Leaderboard

| route | exact_single_crop_expected_BA | mean_score_top4_BA | DEP exact BA | HC exact BA | groups | trials |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_crop_fbstcnet_zscore_only` | 0.695777 | 0.750000 | 0.538347 | 0.774492 | 12 | 96 |

## Crop Performance

| route | crop_0 | crop_1 | crop_2 | crop_3 | crop_4 | best | worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_crop_fbstcnet_zscore_only` | 0.645833 | 0.708333 | 0.708333 | 0.666667 | 0.708333 | crop_1 | crop_0 |

## Calibration

| route | mean | std | min | max | abs_from_0.5 | overconfident_frac | pos_mean | neg_mean | brier | outside_0_1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_crop_fbstcnet_zscore_only` | 0.538163 | 0.188977 | 0.110389 | 0.914334 | 0.158023 | 0.000000 | 0.612446 | 0.463880 | 0.212886 | 0.000000 |

## Score Correlation

| route | `fixed_crop_fbstcnet_zscore_only` |
|---|---:|
| `fixed_crop_fbstcnet_zscore_only` | 1.000000 |

## Mean-score Top-4 Disagreement

| route | `fixed_crop_fbstcnet_zscore_only` |
|---|---:|
| `fixed_crop_fbstcnet_zscore_only` | 0.000000 |

## Frequent Error Samples

| subject | trial | y_true | wrong_models | total_models | models |
|---|---|---:|---:|---:|---|
| `DEP1018` | `DEP1018_neu2` | 0 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `DEP1018` | `DEP1018_neu3` | 0 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `DEP1018` | `DEP1018_neu4` | 0 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `DEP1018` | `DEP1018_pos1` | 1 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `DEP1018` | `DEP1018_pos3` | 1 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `DEP1018` | `DEP1018_pos4` | 1 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `DEP1023` | `DEP1023_neu3` | 0 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `DEP1023` | `DEP1023_neu4` | 0 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `DEP1023` | `DEP1023_pos2` | 1 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `DEP1023` | `DEP1023_pos3` | 1 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `DEP1025` | `DEP1025_neu4` | 0 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `DEP1025` | `DEP1025_pos3` | 1 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `DEP1034` | `DEP1034_neu1` | 0 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `DEP1034` | `DEP1034_neu2` | 0 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `DEP1034` | `DEP1034_pos3` | 1 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `DEP1034` | `DEP1034_pos4` | 1 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `HC1005` | `HC1005_neu2` | 0 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `HC1005` | `HC1005_pos4` | 1 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `HC1011` | `HC1011_neu3` | 0 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `HC1011` | `HC1011_pos2` | 1 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `HC1015` | `HC1015_neu2` | 0 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `HC1015` | `HC1015_pos4` | 1 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `HC1047` | `HC1047_neu3` | 0 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
| `HC1047` | `HC1047_pos2` | 1 | 1 | 1 | `fixed_crop_fbstcnet_zscore_only` |
