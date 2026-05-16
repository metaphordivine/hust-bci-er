# Candidate Score Deep Dive

This report is diagnostic only. It does not change route status and it does not replace candidate or promoted gates.

## Inputs

| route | score_matrix | manifest |
|---|---|---|
| `fixed_crop_ea_fbstcnet` | `outputs/fixed_crop_ea_fbstcnet/candidate_20260516_seed42/score_matrix.csv` | `outputs/fixed_crop_ea_fbstcnet/candidate_20260516_seed42/manifest.json` |

## Leaderboard

| route | exact_single_crop_expected_BA | mean_score_top4_BA | DEP exact BA | HC exact BA | groups | trials |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_fbstcnet` | 0.686144 | 0.750000 | 0.517480 | 0.770476 | 12 | 96 |

## Crop Performance

| route | crop_0 | crop_1 | crop_2 | crop_3 | crop_4 | best | worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_fbstcnet` | 0.625000 | 0.645833 | 0.687500 | 0.708333 | 0.708333 | crop_3 | crop_0 |

## Calibration

| route | mean | std | min | max | abs_from_0.5 | overconfident_frac | pos_mean | neg_mean | brier | outside_0_1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_fbstcnet` | 0.520455 | 0.182946 | 0.132855 | 0.899534 | 0.151086 | 0.000000 | 0.595003 | 0.445907 | 0.209339 | 0.000000 |

## Score Correlation

| route | `fixed_crop_ea_fbstcnet` |
|---|---:|
| `fixed_crop_ea_fbstcnet` | 1.000000 |

## Mean-score Top-4 Disagreement

| route | `fixed_crop_ea_fbstcnet` |
|---|---:|
| `fixed_crop_ea_fbstcnet` | 0.000000 |

## Frequent Error Samples

| subject | trial | y_true | wrong_models | total_models | models |
|---|---|---:|---:|---:|---|
| `DEP1018` | `DEP1018_neu2` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `DEP1018` | `DEP1018_neu3` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `DEP1018` | `DEP1018_neu4` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `DEP1018` | `DEP1018_pos1` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `DEP1018` | `DEP1018_pos3` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `DEP1018` | `DEP1018_pos4` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `DEP1023` | `DEP1023_neu3` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `DEP1023` | `DEP1023_neu4` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `DEP1023` | `DEP1023_pos1` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `DEP1023` | `DEP1023_pos2` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `DEP1025` | `DEP1025_neu1` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `DEP1025` | `DEP1025_pos3` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `DEP1034` | `DEP1034_neu1` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `DEP1034` | `DEP1034_neu2` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `DEP1034` | `DEP1034_pos3` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `DEP1034` | `DEP1034_pos4` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `HC1005` | `HC1005_neu2` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `HC1005` | `HC1005_pos4` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `HC1011` | `HC1011_neu3` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `HC1011` | `HC1011_pos2` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `HC1015` | `HC1015_neu2` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `HC1015` | `HC1015_pos4` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `HC1047` | `HC1047_neu3` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
| `HC1047` | `HC1047_pos1` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet` |
