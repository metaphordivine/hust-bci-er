# Quota Router MoE Component Fusion Best 2026-05-25

Scope: concise repository record for the current highest diagnostic route found
in the active remote campaign. This records the route shape and evidence
anchors only; it does not promote a route, change route status, or commit raw
outputs, checkpoints, prediction tables, or temporary experiment dumps.

## Current Best

- source run: `scratch/quota_router_moe_component_fusion_20260525_1305`
  on F remote worktree
- search family: `fusion_asha_exact`
- kind: `asha_weighted`
- primary metric: `exact_single_crop_expected_BA`
- primary metric value: `0.7551156586666667`
- groups: `300`
- DEP mean BA: `0.66940528`
- HC mean BA: `0.797970848`
- DEP/HC gap: `0.12856556799999996`
- minimum group BA: `0.4190304`

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
- The active best remains `0.7551156586666667` until a later audited run beats
  it.

## Evidence Anchors

Small evidence files were copied locally only for inspection; they remain in
ignored `scratch/` and are not committed.

| remote file | local inspection SHA-256 |
|---|---|
| `manifest.json` | `c2df943054c38abf759e49772b209ce146e81d08bffdd6d47eeb9ba9d4d2fbc2` |
| `fusion_moe_search_report.md` | `4aaec2329de11f63405c3b94aaad094178cf53712116482974aa0b6358df22b3` |
| `fusion_asha_exact.csv` | `1f7b016df70bf67e87f3ade6a6b39380e5aa22f6914f90a12c68ee0d5ae3d6c6` |

## Caveats

- This is diagnostic/posthoc evidence, not candidate or promotion evidence.
- No public/private labels, leaderboard feedback, split changes, or route
  status changes were used.
- The full first-round plan is still incomplete: all ASHA-selected backbones
  must still be converted into aligned score matrices, then the final direct,
  constrained, diversity-aware, and router-MoE fusion passes must be rerun on
  that complete pool.
