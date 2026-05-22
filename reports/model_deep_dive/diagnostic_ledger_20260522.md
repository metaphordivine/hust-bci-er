# Diagnostic Ledger 2026-05-22

## Scope

- Branch context: post-PR38 main.
- Purpose: first-batch diagnostic ledger for EA, whitening eps, sliding-window, DEP/HC balance, and route promotion thresholds.
- Evidence type: summarized protocol metrics only. No checkpoint, raw output, prediction table, public/private label, or split change is used here.
- Route status: unchanged. New routes discussed from this point remain `IDEA` until a separate audit supports promotion.

## Promotion Baseline

The user corrected the lightweight-screen multiplier from `0.92` to `0.98`.

| Symbol | Definition | Current route | Value | Promotion threshold |
|---|---|---|---:|---:|
| `a` | best current `P1 seed42` 5-fold mean | `fixed_crop_car_fbstcnet` | 0.716403 | `0.98a = 0.702075` |
| `b` | best current `P2` eval-7 mean | `fixed_crop_ea_fbstcnet_m_power_light` / `m_conn_light` / `fixed_crop_ea_dgcnn` tie | 0.684524 | `0.98b = 0.670833` |
| `c` | best current `P2 worst` | `tuned_sliding_window_fbstcnet_zscore_only` | 0.520833 | must exceed 0.520833 |

Lightweight IDEA routes should run only `P1 seed42` and `P2` first. A route advances to full experiments if any condition is met:

- `P1 seed42 > 0.702075`
- `P2 eval-7 mean > 0.670833`
- `P2 worst > 0.520833`
- `P1 DEP > 0.82 * P1 HC`
- `P2 DEP > 0.82 * P2 HC`

## First-Batch P2 Evidence

| Route | P2 mean | P2 worst | P2 random | DEP | HC | DEP/HC |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_crop_fbstcnet_zscore_only` | 0.675595 | 0.333333 | 0.770833 | 0.616071 | 0.705357 | 0.873 |
| `fixed_crop_ea_fbstcnet` | 0.672619 | 0.354167 | 0.770833 | 0.598214 | 0.709821 | 0.843 |
| `fixed_crop_whitening_eps3e4_fbstcnet` | 0.660714 | 0.250000 | 0.750000 | 0.571429 | 0.705357 | 0.810 |
| `fixed_crop_whitening_eps1e3_fbstcnet` | 0.657738 | 0.250000 | 0.750000 | 0.562500 | 0.705357 | 0.797 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet` | 0.654762 | 0.270833 | 0.687500 | 0.580357 | 0.691964 | 0.839 |
| `fixed_crop_ea_whitening_eps3e4_fbstcnet` | 0.651786 | 0.291667 | 0.687500 | 0.580357 | 0.687500 | 0.844 |
| `tuned_sliding_window_fbstcnet` | 0.651786 | 0.500000 | 0.687500 | 0.553571 | 0.700893 | 0.790 |
| `tuned_sliding_window_fbstcnet_zscore_only` | 0.648810 | 0.520833 | 0.687500 | 0.589286 | 0.678571 | 0.868 |
| `sliding_window_srfnet_whitening_eps3e4` | 0.648810 | 0.500000 | 0.666667 | 0.642857 | 0.651786 | 0.986 |
| `fixed_crop_deformer_lite_zscore_tuned` | 0.654762 | 0.312500 | 0.791667 | 0.598214 | 0.683036 | 0.876 |
| `sliding_window_deformer_lite` | 0.622024 | 0.437500 | 0.666667 | 0.562500 | 0.651786 | 0.863 |
| `tuned_sliding_window_deformer_lite_zscore_only` | 0.601190 | 0.395833 | 0.645833 | 0.482143 | 0.660714 | 0.730 |
| `fixed_crop_pure_deformer_lite` | 0.595238 | 0.187500 | 0.687500 | 0.482143 | 0.651786 | 0.740 |

## Matched-Pair Deltas

Positive deltas mean the first route in the pair is higher.

| Pair | Delta P2 mean | Delta P2 worst | Delta random | Delta DEP/HC | Read |
|---|---:|---:|---:|---:|---|
| `fixed_crop_ea_whitening_eps3e4_fbstcnet` - `fixed_crop_whitening_eps3e4_fbstcnet` | -0.008928 | +0.041667 | -0.062500 | +0.034 | EA helps worst and DEP/HC under eps3e4, but hurts mean/random. |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet` - `fixed_crop_whitening_eps1e3_fbstcnet` | -0.002976 | +0.020833 | -0.062500 | +0.042 | Same pattern as eps3e4, weaker. |
| `fixed_crop_ea_fbstcnet` - `fixed_crop_fbstcnet_zscore_only` | -0.002976 | +0.020834 | +0.000000 | -0.030 | No global EA gain for fixed FBSTCNet in this matched P2 view. |
| `tuned_sliding_window_fbstcnet_zscore_only` - `fixed_crop_fbstcnet_zscore_only` | -0.026785 | +0.187500 | -0.083333 | -0.005 | Sliding trades mean/random for much better worst-crop robustness. |
| `tuned_sliding_window_deformer_lite_zscore_only` - `fixed_crop_deformer_lite_zscore_tuned` | -0.053572 | +0.083333 | -0.145834 | -0.146 | Sliding helps Deformer worst, but the zscore tuned sliding variant loses too much mean and DEP balance. |

## Diagnostic Conclusions

1. EA is not globally useful by current matched evidence. It may be a stabilizer for worst-crop and DEP/HC under whitening, but it does not currently improve fixed FBSTCNet P2 mean or random crop.

2. Sliding-window is useful as a robustness mechanism, not yet as an overall winner. The strongest signal is `tuned_sliding_window_fbstcnet_zscore_only`: it sets the current `P2 worst` baseline at 0.520833 while staying below top fixed-crop P2 mean.

3. The best fixed-crop routes still look crop-fragile. Several routes score around 0.75-0.79 on random/normal crop but collapse to 0.25-0.35 on worst crop.

4. DEP/HC imbalance is route-specific and cannot be inferred from overall BA. `sliding_window_srfnet_whitening_eps3e4` is not the top P2 mean route, but its P2 DEP/HC ratio is 0.986, making it a useful candidate component for balance-aware fusion or DEP/HC diagnostics.

5. Deformer remains useful as a diversity component. Fixed zscore Deformer has strong random-crop P2, but both fixed and sliding Deformer are still fragile enough that they should not dominate the next standalone backbone design without additional constraints.

## Next Diagnostic Backfill

High-value lightweight backfill before or alongside IDEA routes:

| Priority | Work | Why |
|---:|---|---|
| 1 | `tuned_sliding_window_fbstcnet_zscore_only` P1 seed42 | Completes P1/P2 screen for the current best `P2 worst` route. |
| 2 | `sliding_window_srfnet_whitening_eps3e4` P1 seed42 | Checks whether the strongest DEP/HC-balanced P2 component has any P1 support. |
| 3 | matched P2/P1 seed42 for no-EA and EA sliding FBSTCNet window variants | Separates EA effect from sliding-window shape effect. |
| 4 | DEP/HC exact summary for all future P1 seed42 and P2 IDEA screens | Makes the `DEP > 0.82 * HC` gate auditable from the start. |

## Implications For Today's Exploration

- New backbone designs should combine fixed-crop FBSTCNet strength with explicit crop-robust or balance-aware mechanisms. A pure larger model is not justified by this diagnostic batch.
- Score fusion should treat sliding/SRFNet/Deformer as complementary components, not as direct replacements for fixed FBSTCNet.
- DEP/HC work should be a first-class objective in the lightweight screen, because balanced routes may have lower overall mean but higher downstream value.
