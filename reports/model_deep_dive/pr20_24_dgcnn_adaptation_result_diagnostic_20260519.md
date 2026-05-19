# PR20-24 DGCNN Adaptation Result Diagnostic 2026-05-19

## Scope

- Branch for this report: `codex/pr20-24-result-analysis`
- Upstream PR stack:
  - PR20 `codex/resumable-experiment-runner`: resumable protocol execution, sliding-window routes, official pretrained CBraMod route, P1/P2/P3 runner fixes.
  - PR21 `codex/lightweight-existing-eval`: lightweight existing-evidence evaluation.
  - PR22 `codex/domain-adaptation-ssl`: cohort DANN route for sliding DGCNN.
  - PR23 `codex/cohort-coral-alignment`: cohort CORAL route for sliding DGCNN.
  - PR24 `codex/masked-consistency-ssl`: masked feature-consistency SSL route for sliding DGCNN.
- Remote code root: `/root/autodl-tmp/hust-bci-er-active-fix-20260519`
- Formal P1 root: `outputs/remote_dgcnn_adapt_ssl_p1_repeated_split_20260519_220252`
- Formal P2/P3 root: `outputs/remote_dgcnn_adapt_ssl_fixed_full_20260519_204052`
- Primary metric: `exact_single_crop_expected_BA`
- Status note: this report is diagnostic only. It does not promote route status and does not commit raw outputs, predictions, or checkpoints.

## Completion Evidence

The original fixed-full run had only seed-42 P1, so formal repeated P1 was re-run as a separate split-lane supervisor. The formal evidence used here is:

| Phase | Evidence used | Jobs counted | Audit status | Failure status |
|---|---:|---:|---|---|
| P1 repeated 5x5 | 4 routes x 5 seeds x 5 folds | 100 eval jobs | 100/100 PASS | supervisor returncode 0, no failures/STOP |
| P2 holdout 7-crop | 4 routes x 7 crop policies plus train artifacts | 28 eval jobs + 4 train artifacts | 28/28 eval PASS | protocol results fail 0, skipped 0 |
| P3 nested selection | 4 routes x 45 inner selection + 5 outer final | 180 inner + 20 final | 20/20 final PASS | protocol results fail 0, skipped 0 |

Remote disk after completion was safe at about 35% used with about 66 GB free. Large artifacts were kept remote because night-time high-volume transfer is disallowed.

## P1/P2/P3 Results

| Phase | Rank | Route | n | Mean | Std | Min | Max |
|---|---:|---|---:|---:|---:|---:|---:|
| P1 | 1 | `sliding_ea_dgcnn_masked_consistency_w6_s1` | 25 | 0.640763 | 0.052480 | 0.559460 | 0.737100 |
| P1 | 2 | `sliding_ea_dgcnn_dann_cohort_w6_s1` | 25 | 0.640665 | 0.058999 | 0.539099 | 0.791640 |
| P1 | 3 | `sliding_ea_dgcnn_coral_cohort_w6_s1` | 25 | 0.631911 | 0.055979 | 0.520852 | 0.746833 |
| P1 | 4 | `sliding_ea_dgcnn_w6_s1` | 25 | 0.631622 | 0.057291 | 0.523564 | 0.759040 |
| P2 | 1 | `sliding_ea_dgcnn_coral_cohort_w6_s1` | 7 | 0.705357 | 0.071594 | 0.562500 | 0.791667 |
| P2 | 2 | `sliding_ea_dgcnn_w6_s1` | 7 | 0.687500 | 0.065881 | 0.541667 | 0.729167 |
| P2 | 3 | `sliding_ea_dgcnn_dann_cohort_w6_s1` | 7 | 0.666667 | 0.066970 | 0.520833 | 0.708333 |
| P2 | 4 | `sliding_ea_dgcnn_masked_consistency_w6_s1` | 7 | 0.663690 | 0.065250 | 0.520833 | 0.708333 |
| P3 | 1 | `sliding_ea_dgcnn_dann_cohort_w6_s1` | 5 | 0.648088 | 0.033680 | 0.603717 | 0.698093 |
| P3 | 2 | `sliding_ea_dgcnn_coral_cohort_w6_s1` | 5 | 0.636255 | 0.024336 | 0.607147 | 0.671233 |
| P3 | 3 | `sliding_ea_dgcnn_w6_s1` | 5 | 0.634640 | 0.017107 | 0.622033 | 0.655033 |
| P3 | 4 | `sliding_ea_dgcnn_masked_consistency_w6_s1` | 5 | 0.632790 | 0.032272 | 0.602164 | 0.685667 |

Interpretation:

- PR22 DANN is the strongest P3 route and nearly tied for best P1. This is the most useful PR22-24 addition for final selected-parameter evaluation.
- PR23 CORAL is the strongest P2 route. It improves the pseudo-public holdout/crop stress setting but does not dominate P1/P3.
- PR24 masked consistency gives a tiny P1 mean edge but does not transfer to P2/P3. Treat it as lower priority until a more focused SSL ablation proves otherwise.
- The baseline DGCNN remains competitive, especially in P2/P3, so the adaptation routes are not a clear replacement; they are route-family additions.

## P3 Selection Behavior

| Route | Selected pattern | Diagnosis |
|---|---|---|
| `sliding_ea_dgcnn_w6_s1` | all outer folds selected `input_window_sec=8`, `window_sec=8`, `stride_sec=0.5` | The DGCNN family likely benefits from longer windows under nested selection. The current route name remains `w6_s1`, but P3 final evidence favors 8 s / 0.5 s. |
| `sliding_ea_dgcnn_dann_cohort_w6_s1` | all outer folds selected `adaptation.lambda=0.1` | The tested DANN grid is capped at the selected upper boundary. A follow-up grid should extend above 0.1 before concluding the best strength. |
| `sliding_ea_dgcnn_coral_cohort_w6_s1` | outer folds selected `0.01`, `0.03`, and mostly `0.05` | CORAL is less monotonic than DANN. It has strong P2 behavior, but P3 does not justify a single global lambda without another small grid. |
| `sliding_ea_dgcnn_masked_consistency_w6_s1` | selected `0.02/0.2`, `0.05/0.4`, and `0.1/0.6` across folds | Selection is unstable and final P3 mean is below baseline. Current masked consistency is not yet a reliable SSL mainline. |

## DEP/HC Gap Diagnostic

This diagnostic computes cohort-wise row-level balanced accuracy from each prediction CSV and averages per job. It is not a new official leaderboard metric; it is a route-diagnosis view for the DEP/HC cohort gap.

| Phase | Route | DEP mean | HC mean | Abs gap |
|---|---|---:|---:|---:|
| P1 | `sliding_ea_dgcnn_w6_s1` | 0.471250 | 0.636875 | 0.165625 |
| P1 | `sliding_ea_dgcnn_dann_cohort_w6_s1` | 0.497500 | 0.645000 | 0.147500 |
| P1 | `sliding_ea_dgcnn_coral_cohort_w6_s1` | 0.466250 | 0.642500 | 0.176250 |
| P1 | `sliding_ea_dgcnn_masked_consistency_w6_s1` | 0.477500 | 0.631875 | 0.154375 |
| P2 | `sliding_ea_dgcnn_w6_s1` | 0.513393 | 0.662946 | 0.149554 |
| P2 | `sliding_ea_dgcnn_dann_cohort_w6_s1` | 0.522321 | 0.642857 | 0.120536 |
| P2 | `sliding_ea_dgcnn_coral_cohort_w6_s1` | 0.482143 | 0.683036 | 0.200893 |
| P2 | `sliding_ea_dgcnn_masked_consistency_w6_s1` | 0.500000 | 0.665179 | 0.165179 |
| P3 | `sliding_ea_dgcnn_w6_s1` | 0.512500 | 0.631250 | 0.118750 |
| P3 | `sliding_ea_dgcnn_dann_cohort_w6_s1` | 0.506250 | 0.650000 | 0.143750 |
| P3 | `sliding_ea_dgcnn_coral_cohort_w6_s1` | 0.518750 | 0.631250 | 0.112500 |
| P3 | `sliding_ea_dgcnn_masked_consistency_w6_s1` | 0.487500 | 0.621875 | 0.134375 |

DEP/HC conclusion:

- PR22 DANN reduces the P1 and P2 DEP/HC gap versus baseline, but worsens the P3 gap.
- PR23 CORAL reduces the P3 DEP/HC gap slightly versus baseline, but worsens P1/P2 gap.
- PR24 masked consistency slightly improves P1 gap but worsens P2/P3 gap.
- Therefore PR22-24 do not solve DEP/HC imbalance. They provide useful probes, but the gap remains structurally present and phase-dependent.

## Relation To 2024-2026 EEG Emotion Recognition Trends

The literature trend favors five families: graph/topology, domain adaptation, self-supervised or masked modeling, hybrid CNN/RNN/Transformer, and feature/model fusion. This repo already has credible structure for graph/topology, hybrid backbones, and fusion. The new PR22-24 results confirm that domain adaptation and SSL should become formal route families, but not by simply adding more same-backbone variants.

The next high-value direction is to turn DEP/HC and subject/session shift into explicit diagnostics and route contracts:

- graph-guided alignment, because DGCNN already uses topology but does not yet align graph/domain structure;
- stronger cohort/domain adaptation, because DANN helps P3 score while CORAL helps P2 stress robustness;
- self-supervised pretraining or consistency only if it can show transfer beyond P1 repeated CV.

## Follow-Up Ablations

Priority order:

1. DANN strength extension: run `adaptation.lambda` values above the current selected boundary, for example `0.1`, `0.2`, `0.3`, with full P1/P2/P3. Rationale: P3 selected `0.1` in every outer fold, so the best point may lie outside the current grid.
2. DANN + selected window shape: create a route that combines cohort DANN with the P3-selected DGCNN window shape `8 s / 0.5 s`, then run full P1/P2/P3. Rationale: baseline DGCNN selected 8 s / 0.5 s in every P3 outer fold, while DANN selected the strongest lambda.
3. CORAL high-lambda stress: test whether `0.05` remains best when compared with `0.1` and maybe `0.2`, with special attention to P2 and DEP/HC gap. Rationale: CORAL is P2-best and P3 gap-best, but current selection is not stable enough.
4. DEP/HC diagnostic report tooling: add a small report script that computes cohort-wise BA/gap from protocol outputs and refuses to treat it as route status. Rationale: gap diagnosis should be repeatable without hand-written remote snippets.
5. De-prioritize masked consistency until after a focused SSL redesign. Rationale: current masked consistency does not improve P2/P3 and has unstable P3 selected parameters.

## Working Recommendation

For competition-route diversity, keep all three useful families in the candidate mix:

- score-first route: DANN, because it is P3-best in this DGCNN comparison;
- robustness route: CORAL, because it is P2-best;
- control route: baseline DGCNN or the broader PR20 sliding-window FBSTCNet route, because adaptation gains are not uniform.

Do not spend the next phase only stacking tuned variants of one DGCNN family. The next branch should either extend DANN/CORAL with a small full-protocol ablation or add a reusable DEP/HC cohort-gap diagnostic that can evaluate all future routes.
