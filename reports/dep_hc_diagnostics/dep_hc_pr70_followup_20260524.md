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

## B-only Remote Queue Guidance

Only machine B is available for this follow-up. Do not schedule A/C until the
user explicitly reopens them.

Recommended B queue order:

1. Expand `p01_wide_carz_s7` / `deformer_lite` P2 holdouts to 10-20 seeds.
2. Sweep `balanced_accuracy`, `min_recall`, and `dep_recall_floor_0p8_hc` against `vote_frac`, `mean`, and `trimmed_mean`.
3. Add `temperature` and `platt` calibration only after the uncalibrated sweep is queued.
4. Add `subject_balanced` sampling for the same p01 settings.
5. Run score-level fusion diagnostics from already completed deformer and traditional/time-frequency prediction runs.

Validation requirements:

- Keep output directories under ignored `outputs/`.
- Commit only compact summary CSV/Markdown, never full prediction tables or checkpoints.
- Do not use public/private labels, leaderboard feedback, or split changes.
- Do not use `subject_id`, `trial_id`, filenames, or pseudo IDs as model features.
