# Candidate Score Deep Dive

This report is diagnostic only. It does not change route status and it does not replace candidate or promoted gates.

## Inputs

| route | score_matrix | manifest |
|---|---|---|
| `fixed_crop_ea_cbramod_candidate` | `outputs/fixed_crop_ea_cbramod_candidate/candidate_20260516_seed42/score_matrix.csv` | `outputs/fixed_crop_ea_cbramod_candidate/candidate_20260516_seed42/manifest.json` |

## Leaderboard

| route | exact_single_crop_expected_BA | mean_score_top4_BA | DEP exact BA | HC exact BA | groups | trials |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_cbramod_candidate` | 0.618170 | 0.708333 | 0.510512 | 0.671998 | 12 | 96 |

## Crop Performance

| route | crop_0 | crop_1 | crop_2 | crop_3 | crop_4 | best | worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_cbramod_candidate` | 0.604167 | 0.604167 | 0.666667 | 0.541667 | 0.625000 | crop_2 | crop_3 |

## Calibration

| route | mean | std | min | max | abs_from_0.5 | overconfident_frac | pos_mean | neg_mean | brier | outside_0_1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_cbramod_candidate` | 0.440833 | 0.162388 | 0.146094 | 0.809383 | 0.146407 | 0.000000 | 0.486350 | 0.395316 | 0.234354 | 0.000000 |

## Score Correlation

| route | `fixed_crop_ea_cbramod_candidate` |
|---|---:|
| `fixed_crop_ea_cbramod_candidate` | 1.000000 |

## Mean-score Top-4 Disagreement

| route | `fixed_crop_ea_cbramod_candidate` |
|---|---:|
| `fixed_crop_ea_cbramod_candidate` | 0.000000 |

## Frequent Error Samples

| subject | trial | y_true | wrong_models | total_models | models |
|---|---|---:|---:|---:|---|
| `DEP1018` | `DEP1018_neu2` | 0 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `DEP1018` | `DEP1018_neu3` | 0 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `DEP1018` | `DEP1018_pos3` | 1 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `DEP1018` | `DEP1018_pos4` | 1 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `DEP1023` | `DEP1023_neu3` | 0 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `DEP1023` | `DEP1023_pos2` | 1 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `DEP1025` | `DEP1025_neu1` | 0 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `DEP1025` | `DEP1025_neu4` | 0 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `DEP1025` | `DEP1025_pos2` | 1 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `DEP1025` | `DEP1025_pos3` | 1 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `DEP1034` | `DEP1034_neu1` | 0 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `DEP1034` | `DEP1034_neu2` | 0 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `DEP1034` | `DEP1034_pos3` | 1 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `DEP1034` | `DEP1034_pos4` | 1 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `HC1005` | `HC1005_neu3` | 0 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `HC1005` | `HC1005_pos3` | 1 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `HC1011` | `HC1011_neu3` | 0 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `HC1011` | `HC1011_pos4` | 1 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `HC1013` | `HC1013_neu3` | 0 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `HC1013` | `HC1013_pos2` | 1 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `HC1015` | `HC1015_neu2` | 0 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `HC1015` | `HC1015_pos4` | 1 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `HC1034` | `HC1034_neu3` | 0 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `HC1034` | `HC1034_pos1` | 1 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `HC1039` | `HC1039_neu2` | 0 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `HC1039` | `HC1039_pos3` | 1 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `HC1047` | `HC1047_neu3` | 0 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
| `HC1047` | `HC1047_pos1` | 1 | 1 | 1 | `fixed_crop_ea_cbramod_candidate` |
