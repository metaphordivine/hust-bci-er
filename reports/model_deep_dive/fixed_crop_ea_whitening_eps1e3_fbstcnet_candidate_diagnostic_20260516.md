# Candidate Score Deep Dive

This report is diagnostic only. It does not change route status and it does not replace candidate or promoted gates.

## Inputs

| route | score_matrix | manifest |
|---|---|---|
| `fixed_crop_ea_whitening_eps1e3_fbstcnet` | `outputs/fixed_crop_ea_whitening_eps1e3_fbstcnet/candidate_20260516_seed42/score_matrix.csv` | `outputs/fixed_crop_ea_whitening_eps1e3_fbstcnet/candidate_20260516_seed42/manifest.json` |

## Leaderboard

| route | exact_single_crop_expected_BA | mean_score_top4_BA | DEP exact BA | HC exact BA | groups | trials |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_whitening_eps1e3_fbstcnet` | 0.703641 | 0.770833 | 0.519012 | 0.795956 | 12 | 96 |

## Crop Performance

| route | crop_0 | crop_1 | crop_2 | crop_3 | crop_4 | best | worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_whitening_eps1e3_fbstcnet` | 0.645833 | 0.750000 | 0.708333 | 0.687500 | 0.708333 | crop_1 | crop_0 |

## Calibration

| route | mean | std | min | max | abs_from_0.5 | overconfident_frac | pos_mean | neg_mean | brier | outside_0_1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_crop_ea_whitening_eps1e3_fbstcnet` | 0.504033 | 0.195496 | 0.114448 | 0.860209 | 0.168371 | 0.000000 | 0.583758 | 0.424309 | 0.208511 | 0.000000 |

## Score Correlation

| route | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
|---|---:|
| `fixed_crop_ea_whitening_eps1e3_fbstcnet` | 1.000000 |

## Mean-score Top-4 Disagreement

| route | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
|---|---:|
| `fixed_crop_ea_whitening_eps1e3_fbstcnet` | 0.000000 |

## Frequent Error Samples

| subject | trial | y_true | wrong_models | total_models | models |
|---|---|---:|---:|---:|---|
| `DEP1018` | `DEP1018_neu2` | 0 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `DEP1018` | `DEP1018_neu3` | 0 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `DEP1018` | `DEP1018_neu4` | 0 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `DEP1018` | `DEP1018_pos1` | 1 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `DEP1018` | `DEP1018_pos3` | 1 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `DEP1018` | `DEP1018_pos4` | 1 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `DEP1023` | `DEP1023_neu3` | 0 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `DEP1023` | `DEP1023_neu4` | 0 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `DEP1023` | `DEP1023_pos1` | 1 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `DEP1023` | `DEP1023_pos2` | 1 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `DEP1025` | `DEP1025_neu3` | 0 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `DEP1025` | `DEP1025_pos3` | 1 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `DEP1034` | `DEP1034_neu1` | 0 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `DEP1034` | `DEP1034_neu2` | 0 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `DEP1034` | `DEP1034_pos3` | 1 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `DEP1034` | `DEP1034_pos4` | 1 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `HC1005` | `HC1005_neu2` | 0 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `HC1005` | `HC1005_pos3` | 1 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `HC1011` | `HC1011_neu3` | 0 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `HC1011` | `HC1011_pos2` | 1 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `HC1047` | `HC1047_neu4` | 0 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
| `HC1047` | `HC1047_pos2` | 1 | 1 | 1 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` |
