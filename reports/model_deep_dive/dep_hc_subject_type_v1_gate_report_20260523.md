# DEP/HC Subject-Type V1 Gate Report 2026-05-23

This report is diagnostic-only. It does not change route status, does not
create candidate evidence, and must not be wired into emotion Top-4 routes.

## Scope

- Classifier family: fixed 0.5/0.5 score fusion of `traditional` and
  `time_frequency` DEP/HC feature branches.
- Threshold: fixed `0.5`.
- Subject aggregation: mean crop/window `p_dep`.
- Bucket simulation:
  - HC-like: `p_dep <= 0.35`.
  - DEP-like: `p_dep >= 0.65`.
  - Ambiguous: otherwise.
- Inputs are EEG-derived scores only. `subject_id` is used only for split,
  aggregation, and audit reporting.

## Evidence Files

| artifact | path | purpose |
|---|---|---|
| Baseline board | `reports/model_deep_dive/dep_hc_baseline_board_20260523.md` | Compares traditional, graph, fixed fusion, and validation-selected fusion diagnostics. |
| Hard-subject table | `reports/model_deep_dive/dep_hc_baseline_board_20260523_hard_subjects.csv` | Lists repeated DEP/HC subject-level errors across diagnostic runs. |
| Confidence audit | `reports/model_deep_dive/dep_hc_subject_confidence_audit_20260523.md` | Lists high-confidence subject-type errors and ambiguous counts. |
| Gate simulation | `reports/model_deep_dive/dep_hc_subject_type_gate_sim_20260523.md` | Tests whether subject-type buckets are reliable enough for offline expert-routing simulation. |

## Main Results

| protocol | runs | mean BA | min BA | mean HC recall | mean DEP recall | HC-like precision | DEP-like precision | routable | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P1 | 15 | 0.7958 | 0.6250 | 0.7583 | 0.8333 | 0.9327 | 0.7089 | 0.7222 | fail |
| P2 | 9 | 0.7500 | 0.4375 | 0.6944 | 0.8056 | 0.9026 | 0.7185 | 0.5741 | fail |

The fixed fusion line is still useful as a DEP/HC diagnostic because it has
meaningful cohort signal and often improves over `traditional` alone. It does
not pass the safety gate for expert routing:

- P1 mean BA is just below the 0.80 target and has weak folds at 0.6250.
- P2 has a severe holdout failure at `h666` with subject BA 0.4375.
- DEP-like bucket precision is below 0.75 in both P1 and P2 aggregate views.
- Several high-confidence wrong subjects remain, especially `DEP1003` and
  `DEP1015` being HC-like in weak P1/P2 settings.

## Negative Checks

`graph_connectivity` should not enter the current fusion:

- P2 h123: 0.4375 subject BA.
- P1 fold3: 0.5000 subject BA.

Preprocessing/window changes did not fix the weak split:

- `zscore`-only traditional P2 h123: 0.4375 subject BA.
- `window-sec=5`, `n-crops=10` traditional P2 h123: 0.5625 subject BA.

These checks point to subject-level cohort-atypical behavior rather than a
simple threshold, CAR, or crop-length issue.

## Decision

Do not promote `subject_type_v1` beyond diagnostic use.

Allowed next uses:

- Continue reporting DEP/HC score, confidence, and ambiguous buckets as audit
  metadata.
- Run offline simulations that compare global versus bucket-aware thresholds,
  with no route wiring and no candidate status changes.
- Use the hard-subject list to design new feature families or calibration
  diagnostics.

Disallowed next uses:

- Do not route competition predictions by `p_dep` yet.
- Do not select expert models based on this classifier.
- Do not use subject identifiers as features or memorize hard-subject IDs.

## Next Work Unit

The next high-value local work should be a larger single branch, not small
piecemeal PRs:

1. Build an offline-only bucket-aware threshold simulator.
2. Compare global threshold against HC-like / DEP-like / ambiguous fallbacks.
3. Keep all outputs diagnostic-only and require P1/P2 aggregate gates before
   any expert-router design proceeds.
