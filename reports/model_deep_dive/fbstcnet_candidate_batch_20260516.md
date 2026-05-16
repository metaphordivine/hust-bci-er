# Candidate Score Deep Dive

This report is diagnostic only. It does not change route status and it does not replace candidate or promoted gates.

## Inputs

| route | score_matrix | manifest |
|---|---|---|
| `fixed_crop_ea_fbstcnet` | `outputs/fixed_crop_ea_fbstcnet/candidate_20260516_seed42/score_matrix.csv` | `outputs/fixed_crop_ea_fbstcnet/candidate_20260516_seed42/manifest.json` |
| `fixed_crop_ea_fbstcnet_bands_4_40` | `outputs/fixed_crop_ea_fbstcnet_bands_4_40/candidate_20260516_seed42/score_matrix.csv` | `outputs/fixed_crop_ea_fbstcnet_bands_4_40/candidate_20260516_seed42/manifest.json` |
| `fixed_crop_ea_fbstcnet_c_only` | `outputs/fixed_crop_ea_fbstcnet_c_only/candidate_20260516_seed42/score_matrix.csv` | `outputs/fixed_crop_ea_fbstcnet_c_only/candidate_20260516_seed42/manifest.json` |
| `fixed_crop_ea_fbstcnet_f1_24_f2_48` | `outputs/fixed_crop_ea_fbstcnet_f1_24_f2_48/candidate_20260516_seed42/score_matrix.csv` | `outputs/fixed_crop_ea_fbstcnet_f1_24_f2_48/candidate_20260516_seed42/manifest.json` |
| `fixed_crop_ea_fbstcnet_p_only` | `outputs/fixed_crop_ea_fbstcnet_p_only/candidate_20260516_seed42/score_matrix.csv` | `outputs/fixed_crop_ea_fbstcnet_p_only/candidate_20260516_seed42/manifest.json` |
| `fixed_crop_fbstcnet_zscore_only` | `outputs/fixed_crop_fbstcnet_zscore_only/candidate_20260516_seed42/score_matrix.csv` | `outputs/fixed_crop_fbstcnet_zscore_only/candidate_20260516_seed42/manifest.json` |

## Leaderboard

| route | exact_single_crop_expected_BA | mean_score_top4_BA | DEP exact BA | HC exact BA | groups | trials |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_fbstcnet_c_only` | 0.702109 | 0.750000 | 0.551327 | 0.777500 | 12 | 96 |
| `fixed_crop_fbstcnet_zscore_only` | 0.695777 | 0.750000 | 0.538347 | 0.774492 | 12 | 96 |
| `fixed_crop_ea_fbstcnet_bands_4_40` | 0.694451 | 0.750000 | 0.523524 | 0.779914 | 12 | 96 |
| `fixed_crop_ea_fbstcnet` | 0.686144 | 0.750000 | 0.517480 | 0.770476 | 12 | 96 |
| `fixed_crop_ea_fbstcnet_f1_24_f2_48` | 0.664942 | 0.729167 | 0.491524 | 0.751650 | 12 | 96 |
| `fixed_crop_ea_fbstcnet_p_only` | 0.658259 | 0.729167 | 0.491600 | 0.741588 | 12 | 96 |

## Crop Performance

| route | crop_0 | crop_1 | crop_2 | crop_3 | crop_4 | best | worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_fbstcnet_c_only` | 0.687500 | 0.645833 | 0.687500 | 0.708333 | 0.708333 | crop_3 | crop_1 |
| `fixed_crop_fbstcnet_zscore_only` | 0.645833 | 0.708333 | 0.708333 | 0.666667 | 0.708333 | crop_1 | crop_0 |
| `fixed_crop_ea_fbstcnet_bands_4_40` | 0.708333 | 0.666667 | 0.729167 | 0.645833 | 0.708333 | crop_2 | crop_3 |
| `fixed_crop_ea_fbstcnet` | 0.625000 | 0.645833 | 0.687500 | 0.708333 | 0.708333 | crop_3 | crop_0 |
| `fixed_crop_ea_fbstcnet_f1_24_f2_48` | 0.687500 | 0.604167 | 0.666667 | 0.708333 | 0.645833 | crop_3 | crop_1 |
| `fixed_crop_ea_fbstcnet_p_only` | 0.645833 | 0.645833 | 0.687500 | 0.708333 | 0.666667 | crop_3 | crop_0 |

## Calibration

| route | mean | std | min | max | abs_from_0.5 | overconfident_frac | pos_mean | neg_mean | brier | outside_0_1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_fbstcnet_c_only` | 0.486165 | 0.161593 | 0.121060 | 0.877253 | 0.132569 | 0.000000 | 0.560999 | 0.411330 | 0.201469 | 0.000000 |
| `fixed_crop_fbstcnet_zscore_only` | 0.538163 | 0.188977 | 0.110389 | 0.914334 | 0.158023 | 0.000000 | 0.612446 | 0.463880 | 0.212886 | 0.000000 |
| `fixed_crop_ea_fbstcnet_bands_4_40` | 0.524732 | 0.181101 | 0.142822 | 0.908867 | 0.150409 | 0.000000 | 0.589345 | 0.460119 | 0.218797 | 0.000000 |
| `fixed_crop_ea_fbstcnet` | 0.520455 | 0.182946 | 0.132855 | 0.899534 | 0.151086 | 0.000000 | 0.595003 | 0.445907 | 0.209339 | 0.000000 |
| `fixed_crop_ea_fbstcnet_f1_24_f2_48` | 0.578285 | 0.136749 | 0.253277 | 0.820430 | 0.130505 | 0.000000 | 0.626812 | 0.529758 | 0.226302 | 0.000000 |
| `fixed_crop_ea_fbstcnet_p_only` | 0.491221 | 0.183051 | 0.111044 | 0.919988 | 0.148093 | 0.000000 | 0.550399 | 0.432043 | 0.224407 | 0.000000 |

## Score Correlation

| route | `fixed_crop_ea_fbstcnet` | `fixed_crop_ea_fbstcnet_bands_4_40` | `fixed_crop_ea_fbstcnet_c_only` | `fixed_crop_ea_fbstcnet_f1_24_f2_48` | `fixed_crop_ea_fbstcnet_p_only` | `fixed_crop_fbstcnet_zscore_only` |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_fbstcnet` | 1.000000 | 0.863340 | 0.846168 | 0.933882 | 0.946938 | 0.937461 |
| `fixed_crop_ea_fbstcnet_bands_4_40` | 0.863340 | 1.000000 | 0.867894 | 0.903561 | 0.777240 | 0.923331 |
| `fixed_crop_ea_fbstcnet_c_only` | 0.846168 | 0.867894 | 1.000000 | 0.846671 | 0.695678 | 0.848026 |
| `fixed_crop_ea_fbstcnet_f1_24_f2_48` | 0.933882 | 0.903561 | 0.846671 | 1.000000 | 0.844906 | 0.916299 |
| `fixed_crop_ea_fbstcnet_p_only` | 0.946938 | 0.777240 | 0.695678 | 0.844906 | 1.000000 | 0.878066 |
| `fixed_crop_fbstcnet_zscore_only` | 0.937461 | 0.923331 | 0.848026 | 0.916299 | 0.878066 | 1.000000 |

## Mean-score Top-4 Disagreement

| route | `fixed_crop_ea_fbstcnet` | `fixed_crop_ea_fbstcnet_bands_4_40` | `fixed_crop_ea_fbstcnet_c_only` | `fixed_crop_ea_fbstcnet_f1_24_f2_48` | `fixed_crop_ea_fbstcnet_p_only` | `fixed_crop_fbstcnet_zscore_only` |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_fbstcnet` | 0.000000 | 0.104167 | 0.145833 | 0.104167 | 0.041667 | 0.062500 |
| `fixed_crop_ea_fbstcnet_bands_4_40` | 0.104167 | 0.000000 | 0.125000 | 0.041667 | 0.104167 | 0.062500 |
| `fixed_crop_ea_fbstcnet_c_only` | 0.145833 | 0.125000 | 0.000000 | 0.166667 | 0.145833 | 0.145833 |
| `fixed_crop_ea_fbstcnet_f1_24_f2_48` | 0.104167 | 0.041667 | 0.166667 | 0.000000 | 0.104167 | 0.083333 |
| `fixed_crop_ea_fbstcnet_p_only` | 0.041667 | 0.104167 | 0.145833 | 0.104167 | 0.000000 | 0.104167 |
| `fixed_crop_fbstcnet_zscore_only` | 0.062500 | 0.062500 | 0.145833 | 0.083333 | 0.104167 | 0.000000 |

## Frequent Error Samples

| subject | trial | y_true | wrong_models | total_models | models |
|---|---|---:|---:|---:|---|
| `DEP1018` | `DEP1018_neu2` | 0 | 6 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `DEP1018` | `DEP1018_neu3` | 0 | 6 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `DEP1018` | `DEP1018_pos1` | 1 | 6 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `DEP1018` | `DEP1018_pos3` | 1 | 6 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `DEP1023` | `DEP1023_neu3` | 0 | 6 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `DEP1025` | `DEP1025_pos3` | 1 | 6 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `DEP1034` | `DEP1034_neu1` | 0 | 6 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `DEP1034` | `DEP1034_neu2` | 0 | 6 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `DEP1034` | `DEP1034_pos3` | 1 | 6 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `DEP1034` | `DEP1034_pos4` | 1 | 6 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `HC1005` | `HC1005_neu2` | 0 | 6 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `HC1011` | `HC1011_neu3` | 0 | 6 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `HC1015` | `HC1015_neu2` | 0 | 6 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `HC1015` | `HC1015_pos4` | 1 | 6 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `HC1047` | `HC1047_neu3` | 0 | 6 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `DEP1018` | `DEP1018_neu4` | 0 | 5 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `DEP1018` | `DEP1018_pos4` | 1 | 5 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `DEP1023` | `DEP1023_neu4` | 0 | 5 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `DEP1023` | `DEP1023_pos2` | 1 | 5 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `DEP1023` | `DEP1023_pos3` | 1 | 4 | 6 | `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_fbstcnet_zscore_only` |
| `DEP1025` | `DEP1025_neu1` | 0 | 3 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only` |
| `HC1005` | `HC1005_pos3` | 1 | 3 | 6 | `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_f1_24_f2_48` |
| `HC1005` | `HC1005_pos4` | 1 | 3 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_p_only`, `fixed_crop_fbstcnet_zscore_only` |
| `HC1011` | `HC1011_pos2` | 1 | 3 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_fbstcnet_zscore_only` |
| `HC1011` | `HC1011_pos4` | 1 | 3 | 6 | `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_ea_fbstcnet_p_only` |
| `HC1047` | `HC1047_pos1` | 1 | 3 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_p_only` |
| `HC1047` | `HC1047_pos2` | 1 | 3 | 6 | `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_f1_24_f2_48`, `fixed_crop_fbstcnet_zscore_only` |
| `DEP1023` | `DEP1023_pos1` | 1 | 2 | 6 | `fixed_crop_ea_fbstcnet`, `fixed_crop_ea_fbstcnet_p_only` |
| `DEP1025` | `DEP1025_neu2` | 0 | 2 | 6 | `fixed_crop_ea_fbstcnet_bands_4_40`, `fixed_crop_ea_fbstcnet_c_only` |
| `HC1013` | `HC1013_neu3` | 0 | 2 | 6 | `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_p_only` |
| `HC1013` | `HC1013_pos3` | 1 | 2 | 6 | `fixed_crop_ea_fbstcnet_c_only`, `fixed_crop_ea_fbstcnet_p_only` |
| `DEP1025` | `DEP1025_neu4` | 0 | 1 | 6 | `fixed_crop_fbstcnet_zscore_only` |
| `HC1005` | `HC1005_neu1` | 0 | 1 | 6 | `fixed_crop_ea_fbstcnet_f1_24_f2_48` |
| `HC1005` | `HC1005_pos2` | 1 | 1 | 6 | `fixed_crop_ea_fbstcnet_f1_24_f2_48` |
| `HC1013` | `HC1013_neu4` | 0 | 1 | 6 | `fixed_crop_ea_fbstcnet_c_only` |
| `HC1013` | `HC1013_pos2` | 1 | 1 | 6 | `fixed_crop_ea_fbstcnet_c_only` |
