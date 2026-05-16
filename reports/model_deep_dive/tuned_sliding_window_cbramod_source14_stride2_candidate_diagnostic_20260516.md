# Candidate Score Deep Dive

This report is diagnostic only. It does not change route status and it does not replace candidate or promoted gates.

## Inputs

| route | score_matrix | manifest |
|---|---|---|
| `tuned_sliding_window_cbramod_source14_stride2` | `outputs/tuned_sliding_window_cbramod_source14_stride2/candidate_20260516_seed42/score_matrix.csv` | `outputs/tuned_sliding_window_cbramod_source14_stride2/candidate_20260516_seed42/manifest.json` |

## Leaderboard

| route | exact_single_crop_expected_BA | mean_score_top4_BA | DEP exact BA | HC exact BA | groups | trials |
|---|---:|---:|---:|---:|---:|---:|
| `tuned_sliding_window_cbramod_source14_stride2` | 0.571061 | 0.520833 | 0.473660 | 0.619762 | 12 | 96 |

## Crop Performance

| route | crop_0 | crop_1 | crop_2 | crop_3 | crop_4 | best | worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tuned_sliding_window_cbramod_source14_stride2` | 0.583333 | 0.520833 | 0.520833 | 0.625000 | 0.645833 | crop_4 | crop_1 |

## Calibration

| route | mean | std | min | max | abs_from_0.5 | overconfident_frac | pos_mean | neg_mean | brier | outside_0_1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tuned_sliding_window_cbramod_source14_stride2` | 0.561604 | 0.010110 | 0.545414 | 0.602788 | 0.061604 | 0.000000 | 0.562401 | 0.560807 | 0.253100 | 0.000000 |

## Score Correlation

| route | `tuned_sliding_window_cbramod_source14_stride2` |
|---|---:|
| `tuned_sliding_window_cbramod_source14_stride2` | 1.000000 |

## Mean-score Top-4 Disagreement

| route | `tuned_sliding_window_cbramod_source14_stride2` |
|---|---:|
| `tuned_sliding_window_cbramod_source14_stride2` | 0.000000 |

## Frequent Error Samples

| subject | trial | y_true | wrong_models | total_models | models |
|---|---|---:|---:|---:|---|
| `DEP1018` | `DEP1018_neu1` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1018` | `DEP1018_neu3` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1018` | `DEP1018_pos1` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1018` | `DEP1018_pos3` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1023` | `DEP1023_neu1` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1023` | `DEP1023_neu3` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1023` | `DEP1023_pos2` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1023` | `DEP1023_pos3` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1025` | `DEP1025_neu2` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1025` | `DEP1025_neu3` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1025` | `DEP1025_neu4` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1025` | `DEP1025_pos1` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1025` | `DEP1025_pos3` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1025` | `DEP1025_pos4` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1034` | `DEP1034_neu1` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1034` | `DEP1034_neu2` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1034` | `DEP1034_neu3` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1034` | `DEP1034_pos2` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1034` | `DEP1034_pos3` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `DEP1034` | `DEP1034_pos4` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1005` | `HC1005_neu4` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1005` | `HC1005_pos3` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1011` | `HC1011_neu2` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1011` | `HC1011_neu3` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1011` | `HC1011_neu4` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1011` | `HC1011_pos1` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1011` | `HC1011_pos2` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1011` | `HC1011_pos4` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1013` | `HC1013_neu1` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1013` | `HC1013_neu2` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1013` | `HC1013_neu4` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1013` | `HC1013_pos1` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1013` | `HC1013_pos2` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1013` | `HC1013_pos4` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1015` | `HC1015_neu1` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1015` | `HC1015_pos4` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1033` | `HC1033_neu3` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1033` | `HC1033_pos4` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1034` | `HC1034_neu3` | 0 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
| `HC1034` | `HC1034_pos4` | 1 | 1 | 1 | `tuned_sliding_window_cbramod_source14_stride2` |
