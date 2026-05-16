# Candidate Score Deep Dive

This report is diagnostic only. It does not change route status and it does not replace candidate or promoted gates.

## Inputs

| route | score_matrix | manifest |
|---|---|---|
| `fixed_crop_ea_fbstcnet_m_power_light` | `outputs/fixed_crop_ea_fbstcnet_m_power_light/candidate_20260516_seed42/score_matrix.csv` | `outputs/fixed_crop_ea_fbstcnet_m_power_light/candidate_20260516_seed42/manifest.json` |

## Leaderboard

| route | exact_single_crop_expected_BA | mean_score_top4_BA | DEP exact BA | HC exact BA | groups | trials |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_fbstcnet_m_power_light` | 0.687201 | 0.729167 | 0.520236 | 0.770684 | 12 | 96 |

## Crop Performance

| route | crop_0 | crop_1 | crop_2 | crop_3 | crop_4 | best | worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_fbstcnet_m_power_light` | 0.625000 | 0.666667 | 0.687500 | 0.708333 | 0.729167 | crop_4 | crop_0 |

## Calibration

| route | mean | std | min | max | abs_from_0.5 | overconfident_frac | pos_mean | neg_mean | brier | outside_0_1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_fbstcnet_m_power_light` | 0.531471 | 0.186598 | 0.136677 | 0.901761 | 0.157396 | 0.000000 | 0.607518 | 0.455424 | 0.209763 | 0.000000 |

## Score Correlation

| route | `fixed_crop_ea_fbstcnet_m_power_light` |
|---|---:|
| `fixed_crop_ea_fbstcnet_m_power_light` | 1.000000 |

## Mean-score Top-4 Disagreement

| route | `fixed_crop_ea_fbstcnet_m_power_light` |
|---|---:|
| `fixed_crop_ea_fbstcnet_m_power_light` | 0.000000 |

## Frequent Error Samples

| subject | trial | y_true | wrong_models | total_models | models |
|---|---|---:|---:|---:|---|
| `DEP1018` | `DEP1018_neu2` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `DEP1018` | `DEP1018_neu3` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `DEP1018` | `DEP1018_neu4` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `DEP1018` | `DEP1018_pos1` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `DEP1018` | `DEP1018_pos3` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `DEP1018` | `DEP1018_pos4` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `DEP1023` | `DEP1023_neu3` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `DEP1023` | `DEP1023_neu4` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `DEP1023` | `DEP1023_pos1` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `DEP1023` | `DEP1023_pos2` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `DEP1025` | `DEP1025_neu1` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `DEP1025` | `DEP1025_pos3` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `DEP1034` | `DEP1034_neu1` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `DEP1034` | `DEP1034_neu2` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `DEP1034` | `DEP1034_pos3` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `DEP1034` | `DEP1034_pos4` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `HC1005` | `HC1005_neu2` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `HC1005` | `HC1005_pos4` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `HC1011` | `HC1011_neu3` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `HC1011` | `HC1011_pos4` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `HC1015` | `HC1015_neu2` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `HC1015` | `HC1015_pos4` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `HC1033` | `HC1033_neu2` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `HC1033` | `HC1033_pos3` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `HC1047` | `HC1047_neu3` | 0 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
| `HC1047` | `HC1047_pos1` | 1 | 1 | 1 | `fixed_crop_ea_fbstcnet_m_power_light` |
