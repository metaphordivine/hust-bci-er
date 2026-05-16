# Candidate Score Deep Dive

This report is diagnostic only. It does not change route status and it does not replace candidate or promoted gates.

## Inputs

| route | score_matrix | manifest |
|---|---|---|
| `fixed_crop_ea_deformer_mean_pool_control` | `outputs/fixed_crop_ea_deformer_mean_pool_control/candidate_20260516_seed42/score_matrix.csv` | `outputs/fixed_crop_ea_deformer_mean_pool_control/candidate_20260516_seed42/manifest.json` |

## Leaderboard

| route | exact_single_crop_expected_BA | mean_score_top4_BA | DEP exact BA | HC exact BA | groups | trials |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_deformer_mean_pool_control` | 0.659829 | 0.750000 | 0.573020 | 0.703234 | 12 | 96 |

## Crop Performance

| route | crop_0 | crop_1 | crop_2 | crop_3 | crop_4 | best | worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_deformer_mean_pool_control` | 0.625000 | 0.604167 | 0.708333 | 0.750000 | 0.666667 | crop_3 | crop_1 |

## Calibration

| route | mean | std | min | max | abs_from_0.5 | overconfident_frac | pos_mean | neg_mean | brier | outside_0_1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_deformer_mean_pool_control` | 0.547334 | 0.180365 | 0.182455 | 0.909883 | 0.160330 | 0.000000 | 0.610830 | 0.483837 | 0.221275 | 0.000000 |

## Score Correlation

| route | `fixed_crop_ea_deformer_mean_pool_control` |
|---|---:|
| `fixed_crop_ea_deformer_mean_pool_control` | 1.000000 |

## Mean-score Top-4 Disagreement

| route | `fixed_crop_ea_deformer_mean_pool_control` |
|---|---:|
| `fixed_crop_ea_deformer_mean_pool_control` | 0.000000 |

## Frequent Error Samples

| subject | trial | y_true | wrong_models | total_models | models |
|---|---|---:|---:|---:|---|
| `DEP1018` | `DEP1018_neu2` | 0 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `DEP1018` | `DEP1018_neu4` | 0 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `DEP1018` | `DEP1018_pos1` | 1 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `DEP1018` | `DEP1018_pos3` | 1 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `DEP1023` | `DEP1023_neu2` | 0 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `DEP1023` | `DEP1023_neu4` | 0 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `DEP1023` | `DEP1023_pos1` | 1 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `DEP1023` | `DEP1023_pos3` | 1 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `DEP1025` | `DEP1025_neu4` | 0 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `DEP1025` | `DEP1025_pos2` | 1 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `DEP1034` | `DEP1034_neu1` | 0 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `DEP1034` | `DEP1034_neu2` | 0 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `DEP1034` | `DEP1034_pos3` | 1 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `DEP1034` | `DEP1034_pos4` | 1 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `HC1005` | `HC1005_neu2` | 0 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `HC1005` | `HC1005_pos3` | 1 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `HC1011` | `HC1011_neu3` | 0 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `HC1011` | `HC1011_pos2` | 1 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `HC1013` | `HC1013_neu4` | 0 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `HC1013` | `HC1013_pos2` | 1 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `HC1015` | `HC1015_neu2` | 0 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `HC1015` | `HC1015_pos4` | 1 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `HC1034` | `HC1034_neu2` | 0 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
| `HC1034` | `HC1034_pos4` | 1 | 1 | 1 | `fixed_crop_ea_deformer_mean_pool_control` |
