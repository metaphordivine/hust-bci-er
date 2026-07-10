# DEP/HC PR70 B Score Fusion Diagnostic 2026-05-24

Diagnostic-only: results are from machine B PR70 outputs and P3 outer-final labels. They are not emotion-route candidate evidence and do not change route status.

## Best Fusion Strategies

| strategy | components | expected BA mean | expected BA min | worst BA min | min recall mean | sort BA mean | note |
|---|---|---:|---:|---:|---:|---:|---|
| `tsc_f8_toptsc8_equal_fixed05` | `tsc_f8+top_tsc8` | 0.8141 | 0.6652 | 0.4375 | 0.7074 | 0.8125 | best mean, fixed 0.5 threshold |
| `p01_tsc_f8_toptsc8_equal_balacc` | `p01+tsc_f8+top_tsc8` | 0.8036 | 0.6537 | 0.3750 | 0.6780 | 0.7750 | inner-threshold tuned; check overfit risk |
| `tsc_f8_toptsc8_equal_balacc` | `tsc_f8+top_tsc8` | 0.7989 | 0.6426 | 0.3750 | 0.6778 | 0.7750 | inner-threshold tuned; check overfit risk |
| `p01_tsc_f8_tsc4_equal_balacc` | `p01+tsc_f8+tsc_f4` | 0.7913 | 0.6071 | 0.3750 | 0.6953 | 0.7750 | inner-threshold tuned; check overfit risk |
| `p01_tsc_f8_toptsc8_equal_fixed05` | `p01+tsc_f8+top_tsc8` | 0.7908 | 0.6253 | 0.4375 | 0.6530 | 0.7750 |  |
| `p01_tsc_f8_equal_fixed05` | `p01+tsc_f8` | 0.7820 | 0.5350 | 0.4375 | 0.6511 | 0.8500 | best known-count sort mean among simple p01+tsc pair |
| `p01_tsc_f8_tsc4_equal_fixed05` | `p01+tsc_f8+tsc_f4` | 0.7784 | 0.5761 | 0.3750 | 0.6426 | 0.7750 |  |
| `p01_tsc_f8_conv32_equal_fixed05` | `p01+tsc_f8+def_conv32` | 0.7774 | 0.5662 | 0.5000 | 0.6601 | 0.8125 |  |

## Single-Model Outer-Final Baselines

| component | model | expected BA mean | expected BA min | worst BA min | min recall mean | sort BA mean |
|---|---|---:|---:|---:|---:|---:|
| `tsc_f8` | `tsception` | 0.7801 | 0.5335 | 0.3125 | 0.6670 | 0.8500 |
| `p01` | `deformer_lite` | 0.7567 | 0.5769 | 0.3750 | 0.5790 | 0.8500 |
| `def_conv32` | `deformer_lite` | 0.7498 | 0.5876 | 0.4375 | 0.5920 | 0.8125 |
| `top_tsc8` | `tsception` | 0.7367 | 0.5777 | 0.4375 | 0.5590 | 0.7750 |
| `def_conv24` | `deformer_lite` | 0.7223 | 0.5335 | 0.3750 | 0.5378 | 0.7750 |
| `tsc_f4` | `tsception` | 0.7121 | 0.5108 | 0.4375 | 0.5304 | 0.7750 |
| `tri_context` | `tri_context_gate` | 0.6969 | 0.6376 | 0.3750 | 0.5055 | 0.7375 |

## Interpretation

- The strongest simple fusion is `tsc_f8_toptsc8_equal_fixed05`: expected BA mean 0.8141, higher than the best single outer-final baseline in this table (`tsc_f8`, 0.7801).
- The fixed-threshold pair `p01_tsc_f8_equal_fixed05` remains useful because known-count sorting reaches 0.8500 mean and fixed threshold avoids inner-threshold overfit.
- Inner-threshold tuning improves some means but often lowers worst-combo minimum; do not promote it without another P3-style confirmation or more holdout seeds.
- Keep all outputs under ignored `outputs/`; only these compact summaries are suitable for PR70.

Artifacts:
- `reports/dep_hc_diagnostics/dep_hc_pr70_score_fusion_20260524_summary.tsv`
- `reports/dep_hc_diagnostics/dep_hc_pr70_score_fusion_20260524_by_outer.tsv`
- `reports/dep_hc_diagnostics/dep_hc_pr70_score_fusion_20260524_single_outer_summary.tsv`
- `reports/dep_hc_diagnostics/dep_hc_pr70_score_fusion_20260524_single_outer_by_outer.tsv`
