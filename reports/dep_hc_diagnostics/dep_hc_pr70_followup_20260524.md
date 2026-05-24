# DEP/HC PR70 Follow-up Plan 2026-05-24

This note is diagnostic-only. It does not change route status and is not
candidate evidence for the emotion Top-4 route board.

## Default Candidate Policy

- Default DEP/HC diagnostic candidate: `p01_wide_carz_s7` / `deformer_lite` / `balanced_accuracy` + `vote_frac`.
- Reason: `min_recall/vote_frac` has the higher P2 mean BA, but `balanced_accuracy/vote_frac` has better worst-combo BA and a smaller HC/DEP recall imbalance.
- Primary comparison remains `crop_combo_expected_*` when available. Legacy all-crop subject metrics remain audit-only.
- Ranking for DEP/HC defaults should prefer recall balance and worst-combo stability before raw mean BA.

## Implemented Support

- Neural diagnostic runs now support `--calibration-method none|temperature|platt|isotonic`.
- Neural diagnostic runs now support `--sampling-strategy window|subject_balanced`.
- Route-to-DEP/HC planning can expand threshold-objective by subject-aggregation sweeps in one TSV.
- DEP/HC score-level fusion can consume arbitrary existing `dep_hc_predictions.csv` component runs.
- Summary boards now expose `min(HC, DEP)` recall, recall gap, mean worst-combo BA, and hard-subject false-negative/false-positive split.
- DEP/HC diagnostic runners now support P3 outer/inner splits for tuning confirmation. P3 inner folds are used for selection; outer-final folds are held out for confirmation.

## P3 Fusion Confirmation

Remote B ran 60 P3 component diagnostics for `p01_wide_carz_s7` / `deformer_lite`:

- components: `win_temp`, `win_platt`, `sub_temp`;
- split: 5 P3 outer folds, 3 inner folds per outer, plus one outer-final run per component/fold;
- execution: 5 GPU lanes x 3 CPU threads per lane during training; P3 fusion summary rerun with 15 CPU worker processes;
- result artifacts: compact TSVs only, under `reports/dep_hc_diagnostics/dep_hc_pr70_p3_quota_fusion_20260524_*.tsv`.

P3 outer-final strategy summary:

| Strategy | Weights | Sort mean | Sort min | Subject BA mean | Crop-combo mean | Crop worst min |
|---|---|---:|---:|---:|---:|---:|
| `p3_global_inner_selected` | `win_temp=0.00, win_platt=0.50, sub_temp=0.50` | 0.8125 | 0.6250 | 0.7625 | 0.7590 | 0.5000 |
| `p2_w0p25_0p15_0p60` | `win_temp=0.25, win_platt=0.15, sub_temp=0.60` | 0.8125 | 0.6250 | 0.7625 | 0.7558 | 0.5000 |
| `p2_w0p20_0p20_0p60` | `win_temp=0.20, win_platt=0.20, sub_temp=0.60` | 0.8125 | 0.6250 | 0.7500 | 0.7530 | 0.5000 |
| `equal` | `win_temp=0.33, win_platt=0.33, sub_temp=0.33` | 0.8125 | 0.6250 | 0.7625 | 0.7488 | 0.5000 |
| `p3_per_outer_inner_selected` | mixed per outer | 0.8125 | 0.6250 | 0.7250 | 0.7270 | 0.5000 |

Interpretation:

- Known DEP-count sorting is saturated across these simple 3-component fusions on P3 outer-final: all strategies have `sort_mean=0.8125` and `sort_min=0.6250`.
- P3 inner-selected global weights (`0.00/0.50/0.50`) give the best crop-combo mean among the tested simple fusions, but the margin over the P2-selected `0.25/0.15/0.60` is small.
- Per-outer inner selection overfits and should not be used as the default.
- Current default for further DEP/HC score-fusion diagnostics: compare `0.00/0.50/0.50` and `0.25/0.15/0.60`; keep equal weights as the sanity baseline.

## B-only Remote Queue Guidance

Only machine B is available for this follow-up. Do not schedule A/C until the
user explicitly reopens them.

Recommended B queue order:

1. Expand `p01_wide_carz_s7` / `deformer_lite` P2 holdouts to 10-20 seeds.
2. Sweep `balanced_accuracy`, `min_recall`, and `dep_recall_floor_0p8_hc` against `vote_frac`, `mean`, and `trimmed_mean`.
3. Add `temperature` and `platt` calibration only after the uncalibrated sweep is queued.
4. Add `subject_balanced` sampling for the same p01 settings.
5. Run score-level fusion diagnostics from already completed deformer and traditional/time-frequency prediction runs.
6. Use P3 inner folds for any new threshold/weight tuning claim, then report P3 outer-final results separately.

Validation requirements:

- Keep output directories under ignored `outputs/`.
- Commit only compact summary CSV/Markdown, never full prediction tables or checkpoints.
- Do not use public/private labels, leaderboard feedback, or split changes.
- Do not use `subject_id`, `trial_id`, filenames, or pseudo IDs as model features.
