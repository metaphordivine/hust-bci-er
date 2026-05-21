# Riemannian-Gated FBSTCNet P1 Protocol Report

Date: 2026-05-20

Branch: `codex/fbstcnet-riem-gate`
Route: `fixed_crop_ea_fbstcnet_riem_guided_gate`
Remote worktree: `/root/autodl-tmp/hust-bci-er-fbstcnet-riem-gate-20260520-1958`
Remote run root: `outputs/remote_fbstcnet_riem_gate_p1_full_20260520_2000`

## Result

The P1 repeated group-kfold run completed all 25 candidate jobs.

| scope | result |
|---|---:|
| seeds | `42`, `123`, `456`, `789`, `1024` |
| folds per seed | 5 |
| completed candidate jobs | 25 |
| primary metric | `exact_single_crop_expected_BA` |
| mean | 0.6883451306666667 |
| population std | 0.04606313161271889 |
| min | 0.591428 |
| max | 0.7764253333333334 |
| initial audit status | `BLOCKED` only by missing route summary |
| final protocol rerun | `failed_jobs=0` |

The route is a competitive fixed-crop model-code candidate and is clearly stronger than the standalone Riemannian tangent baseline. It is still below the current best P1 score-fusion runs, so it should be treated as a strong component or backbone candidate until P2/P3 evidence is available.

## Gate Notes

All 25 job-level candidate audits failed only `SUMMARY_EXISTS` during the first execution pass:

```text
route summary is missing: reports/route_summaries/fixed_crop_ea_fbstcnet_riem_guided_gate_summary.md
```

The inspected audit reports passed route schema, config provenance, git commit containment, candidate mode, genuine score matrix evidence, dataset checksum evidence, split disjointness, test-only prediction scope, metric recomputation, and route metric consistency.

After the summary was present, the candidate audit passed against an individual job and the full protocol command skipped all 25 existing artifacts with `failed_jobs=0`.

## Interpretation

The Riemannian gate improves the usefulness of the FBSTCNet-M mixed branch by letting covariance/tangent features choose between power and connectivity branch probabilities. The P1 mean of 0.6883451306666667 is high enough to keep the branch-gated backbone in the main candidate pool, but it is not promotion evidence.

The most useful next checks are:

- P2 crop-stress evaluation to see whether the learned branch gate is stable under crop policy changes.
- P3 nested evaluation to test whether the gate remains useful when the selection path cannot see the final test folds.
- Branch-weight diagnostics by subject/cohort and crop position, using only held-out predictions and without using subject identifiers as model features.

## Follow-Up

- Keep the route status as `IDEA` until P2/P3 evidence exists.
- Compare branch-gate confidence against fixed equal-probability averaging to verify that the gate is not just adding variance.
- Use the route as a diversity component for score fusion only after genuine P2/P3 component score matrices are available.
