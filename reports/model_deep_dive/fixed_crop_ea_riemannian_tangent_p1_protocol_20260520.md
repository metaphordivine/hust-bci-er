# Riemannian Tangent P1 Protocol Report

Date: 2026-05-20

Branch: `codex/route-riemannian-component`
Route: `fixed_crop_ea_riemannian_tangent`
Remote worktree: `/root/autodl-tmp/hust-bci-er-route-riemannian-component-20260520-1820`
Remote run root: `outputs/remote_riemannian_tangent_p1_full_20260520_1820`

## Result

The P1 repeated group-kfold run completed all 25 candidate jobs.

| scope | result |
|---|---:|
| seeds | `42`, `123`, `456`, `789`, `1024` |
| folds per seed | 5 |
| completed candidate jobs | 25 |
| primary metric | `exact_single_crop_expected_BA` |
| mean | 0.5943390506666667 |
| population std | 0.03347838151830074 |
| min | 0.5184951999999999 |
| max | 0.6618464 |
| initial audit status | `BLOCKED` only by missing route summary |

The route is a useful low-variance covariance/tangent-space baseline, but it is not competitive with the stronger CNN, graph, and score-fusion routes from this campaign. Its main value is interpretability and as a future fusion component, not as a standalone promotion candidate.

## Gate Notes

All 25 job-level candidate audits failed only `SUMMARY_EXISTS` during the first execution pass:

```text
route summary is missing: reports/route_summaries/fixed_crop_ea_riemannian_tangent_summary.md
```

The train/evidence checks passed on the inspected job reports, including route schema, config provenance, git commit containment, candidate mode, genuine score matrix evidence, dataset checksum evidence, split disjointness, test-only prediction scope, and route metric consistency.

After the summary is committed, rerun the candidate audit against an individual job or rerun the protocol command over the existing artifacts so the runner can skip completed jobs and refresh audit status.

## Interpretation

The log-covariance tangent model underfits compared with current best routes. That is expected: the architecture intentionally keeps a small linear head over covariance tangent features. The result still gives two useful pieces of evidence:

- The SPD covariance and tangent-space path is runnable under the repository's formal candidate protocol.
- The route creates a clean basis for channel-pair coefficient interpretability and for diverse score fusion with CNN/transformer/graph families.

## Follow-Up

- Add tangent coefficient and channel-pair attribution export as a lightweight interpretability artifact.
- Use the route as a diversity component only after genuine component score matrices are aligned with the target fusion protocol.
- Do not promote the standalone route without P2/P3 evidence and a clear benefit over stronger baselines.
