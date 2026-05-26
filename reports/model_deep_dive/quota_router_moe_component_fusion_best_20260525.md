# Quota Router MoE Component Fusion Best 2026-05-25

Scope: concise repository record for the current highest diagnostic
score-fusion record found in the active remote campaign. This records the
fusion shape and evidence anchors only; it is not a registered route config,
does not promote a route, does not change route status, and does not commit raw
outputs, checkpoints, prediction tables, or temporary experiment dumps.

## Current Best

| field | value |
|---|---|
| record status | diagnostic score-fusion record only |
| route registry status | not registered as a route config |
| route status change | none |
| source run | `scratch/quota_router_moe_component_fusion_20260525_1305` on F remote worktree |
| search family | `fusion_asha_exact` |
| search kind | `asha_weighted` |
| primary metric | `exact_single_crop_expected_BA` |
| primary metric value | `0.7551156586666667` |
| evaluation groups | `300` |
| DEP mean BA | `0.66940528` |
| HC mean BA | `0.797970848` |
| DEP/HC gap | `0.12856556799999996` |
| minimum group BA | `0.4190304` |

## Method

The search assembled a diagnostic score-fusion route from already-produced
component and virtual-MoE score artifacts. It did not train a new backbone in
this PR and did not create a new registered route config.

The method was:

1. Start from the active remote score-fusion pool for the 2026-05-25 campaign.
2. Evaluate candidate component mixtures with the `fusion_asha_exact` family.
3. Use ASHA-style successive halving to spend more evaluation on promising
   weighted mixtures while pruning weaker mixtures.
4. Score each candidate with `exact_single_crop_expected_BA` over 300 groups.
5. Keep the best diagnostic weighted fusion found in the 13:05 run.
6. Compare it against the best non-ASHA weighted-pair result from the same run
   and the later imported 17:05 diagnostic run.

The virtual-MoE components in the winning mixture are hard quota-router
diagnostics. Their component names record the DEP expert, HC expert, transform,
and quota router family used by the source search. They are evidence anchors
for this diagnostic result, not inference-time use of public/private labels,
leaderboard feedback, subject identifiers, trial identifiers, filenames, or any
route-status promotion.

Promotion to a registered route would require a separate PR that adds an
auditable config under `configs/routes/models/`, updates
`reports/route_registry.yaml` without changing status beyond supported evidence,
and passes the relevant experiment gate. This PR does not do that.

Best weighted components:

| component | weight |
|---|---:|
| `expert:fixed_crop_car_fbstcnet_robust_clip_train_mad8_v1_hc_w2_focal20` | `0.541` |
| `virtual_moe:p01_tsc_f8_equal_depfloor:quota_hard` with DEP `fixed_crop_car_fbstcnet_robust_clip_train_mad8_v1_dep_w2_focal20`, HC `assembled__srfnet_reference_context__nonfrontal_channel_dropout_v1`, transform `global_z` | `0.151` |
| `virtual_moe:p01_tsc_f8_select2_depfloor:quota_hard` with DEP `fixed_crop_car_fbstcnet_robust_clip_train_mad8_v1_dep_w2_focal20`, HC `assembled__srfnet_reference_context__robust_clip_train_mad8_v1`, transform `sigmoid_t1p25` | `0.080` |
| `virtual_moe:p01_tsc_f8_select2_depfloor:quota_hard` with DEP `fixed_crop_car_fbstcnet_train_channel_zscore_v1_dep_w2_focal20`, HC `assembled__srfnet_reference_context__robust_clip_train_mad8_v1`, transform `global_z` | `0.184` |
| `assembled__noea_mconn_context__nonfrontal_channel_dropout_v1` | `0.043` |

## Comparison

- Best non-ASHA weighted-pair result in the same 13:05 run:
  `0.7544274453333334`.
- Later imported diagnostic run at
  `scratch/quota_router_moe_component_fusion_20260525_1705_imported` reached
  `0.7549376533333333`, below the 13:05 best.
- The active diagnostic best remains `0.7551156586666667` until a later
  audited run beats it.

## Evidence Anchors

Small evidence files were copied locally only for inspection; they remain in
ignored `scratch/` and are not committed. The large binary handoff artifacts
for the campaign are published as release assets instead of git-tracked files:

```text
https://github.com/metaphordivine/hust-bci-er/releases/tag/v2026.05.26-remote-artifacts
```

| remote file | local inspection SHA-256 |
|---|---|
| `manifest.json` | `c2df943054c38abf759e49772b209ce146e81d08bffdd6d47eeb9ba9d4d2fbc2` |
| `fusion_moe_search_report.md` | `4aaec2329de11f63405c3b94aaad094178cf53712116482974aa0b6358df22b3` |
| `fusion_asha_exact.csv` | `1f7b016df70bf67e87f3ade6a6b39380e5aa22f6914f90a12c68ee0d5ae3d6c6` |

## Reproduction Boundary

This PR is intended to make the diagnostic result auditable, not to claim a
complete reproducible candidate run. To inspect the archived artifacts:

```text
gh release download v2026.05.26-remote-artifacts --pattern SHA256SUMS_GM_20260526.txt
gh release download v2026.05.26-remote-artifacts --pattern "*_essential_pull_20260526.tar"
sha256sum -c SHA256SUMS_GM_20260526.txt --ignore-missing
```

The 13:05 diagnostic run remains the source of the route, score, and method
recorded here. A full candidate-level reproduction would require rerunning the
registered protocol from committed route configs and keeping route status
unchanged until the experiment gate passes.

## Caveats

- This is diagnostic/posthoc evidence, not candidate or promotion evidence.
- No public/private labels, leaderboard feedback, split changes, or route
  status changes were used.
- The full first-round plan is still incomplete: all ASHA-selected backbones
  must still be converted into aligned score matrices, then the final direct,
  constrained, diversity-aware, and router-MoE fusion passes must be rerun on
  that complete pool.
