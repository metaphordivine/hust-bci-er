# DEP/HC PR70 Subagent Follow-up Fusion 2026-05-24

Diagnostic-only follow-up based on the DEP/HC exploration subagent recommendations. Results use existing B-machine PR70 prediction artifacts and P3 outer-final evaluation; no route status is changed.

## Summary

| strategy | components | expected BA mean | expected BA min | worst BA min | min recall mean | DEP/HC mean | sort BA mean | recall floor outer count | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `tsc_pair_p01_w45_45_10_fixed05` | `tsc_f8+top_tsc8+p01` | 0.8119 | 0.6668 | 0.4375 | 0.7041 | 0.8643 | 0.8125 | 3/5 |  |
| `tsc_pair_p01_w40_40_20_fixed05` | `tsc_f8+top_tsc8+p01` | 0.8104 | 0.6608 | 0.4375 | 0.6931 | 0.8643 | 0.8125 | 3/5 |  |
| `tsc_pair_def32_w45_45_10_fixed05` | `tsc_f8+top_tsc8+def_conv32` | 0.8089 | 0.6656 | 0.3750 | 0.6989 | 0.8643 | 0.8125 | 3/5 |  |
| `tsc_pair_tri_w40_40_20_fixed05` | `tsc_f8+top_tsc8+tri_context` | 0.8055 | 0.6778 | 0.4375 | 0.6891 | 0.9024 | 0.8125 | 3/5 |  |
| `tsc_pair_tri_w45_45_10_fixed05` | `tsc_f8+top_tsc8+tri_context` | 0.8045 | 0.6775 | 0.4375 | 0.6907 | 0.8643 | 0.8125 | 3/5 |  |
| `tsc_pair_def32_w40_40_20_fixed05` | `tsc_f8+top_tsc8+def_conv32` | 0.7965 | 0.6468 | 0.3750 | 0.6858 | 0.9024 | 0.8125 | 3/5 | more deformer weight improves DEP/HC ratio but hurts worst-min |
| `p01_tsc_f8_equal_depfloor` | `p01+tsc_f8` | 0.7261 | 0.5462 | 0.3750 | 0.5326 | 1.6248 | 0.8125 | 4/5 |  |
| `p01_tsc_f8_select2_depfloor` | `p01+tsc_f8` | 0.7018 | 0.5128 | 0.3125 | 0.5189 | 1.8781 | 0.8125 | 5/5 | explicit DEP floor selection, watch mean drop |

## Interpretation

- The best previous baseline remains `tsc_f8_toptsc8_equal_fixed05` from `dep_hc_pr70_score_fusion_20260524_summary.tsv` (`expected BA mean=0.8141`).
- This follow-up tests whether deformer or tri-context small weights can improve worst-combo stability, and whether DEP-floor inner selection can reduce recall imbalance.
- The best near-baseline follow-up is `tsc_pair_p01_w45_45_10_fixed05`: expected BA stays close to the previous top result while mean DEP/HC ratio improves from 0.8571 to 0.8643; it does not solve the worst-min issue.
- Explicit DEP-floor selection passes the recall-floor check more often, but drops expected BA too much to use as the default.
- Promotion requires improving worst-min or DEP/HC balance without materially dropping expected BA; otherwise keep the result diagnostic-only.

Artifacts:
- `reports/dep_hc_diagnostics/dep_hc_pr70_subagent_followup_fusion_20260524_summary.tsv`
- `reports/dep_hc_diagnostics/dep_hc_pr70_subagent_followup_fusion_20260524_by_outer.tsv`
