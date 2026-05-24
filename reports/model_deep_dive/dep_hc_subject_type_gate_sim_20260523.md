# DEP/HC Subject-Type Gate Simulation

This report is diagnostic-only. It must not be used as candidate evidence or wired into competition routes.

Subject-type buckets: HC-like `p_dep <= 0.35`, DEP-like `p_dep >= 0.65`, otherwise ambiguous.

## Aggregate Gate

| protocol | runs | mean BA | min BA | mean HC recall | mean DEP recall | HC-like precision | DEP-like precision | routable | pass | failed reasons |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| p1 | 15 | 0.7958 | 0.6250 | 0.7583 | 0.8333 | 0.9327 | 0.7089 | 0.7222 | false | mean_ba;hc_like_precision;dep_like_precision;run_level |
| p2 | 9 | 0.7500 | 0.4375 | 0.6944 | 0.8056 | 0.9026 | 0.7185 | 0.5741 | false | mean_ba;min_ba;hc_recall;hc_like_precision;dep_like_precision;run_level |

## Run Gate

| run | protocol | BA | HC recall | DEP recall | HC-like | DEP-like | ambiguous | HC-like precision | DEP-like precision | pass | failed reasons |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h123_w050 | p2 | 0.6875 | 0.6250 | 0.7500 | 3 | 2 | 7 | 0.6667 | 0.5000 | false | hc_like_precision;dep_like_precision |
| dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h456_w050 | p2 | 0.8125 | 0.8750 | 0.7500 | 7 | 4 | 1 | 0.8571 | 0.7500 | true |  |
| dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h999_w050 | p2 | 0.7500 | 0.5000 | 1.0000 | 3 | 1 | 8 | 1.0000 | 1.0000 | true |  |
| dep_hc_subject_type_v1_20260523_2120_p1_fold0_fixed_w050 | p1 | 0.8125 | 0.6250 | 1.0000 | 3 | 5 | 4 | 1.0000 | 0.6000 | false | dep_like_precision |
| dep_hc_subject_type_v1_20260523_2120_p1_fold1_fixed_w050 | p1 | 0.8750 | 0.7500 | 1.0000 | 3 | 5 | 4 | 1.0000 | 0.8000 | true |  |
| dep_hc_subject_type_v1_20260523_2120_p1_fold2_fixed_w050 | p1 | 0.9375 | 0.8750 | 1.0000 | 5 | 4 | 3 | 1.0000 | 0.7500 | true |  |
| dep_hc_subject_type_v1_20260523_2120_p1_fold3_fixed_w050 | p1 | 0.6250 | 0.7500 | 0.5000 | 6 | 3 | 3 | 0.6667 | 0.3333 | false | hc_like_precision;dep_like_precision |
| dep_hc_subject_type_v1_20260523_2120_p1_fold4_fixed_w050 | p1 | 0.7500 | 0.7500 | 0.7500 | 6 | 3 | 3 | 1.0000 | 0.6667 | false | dep_like_precision |
| dep_hc_subject_type_v1_20260523_2120_p2_h1024_fixed_w050 | p2 | 0.8750 | 0.7500 | 1.0000 | 5 | 4 | 3 | 1.0000 | 0.7500 | true |  |
| dep_hc_subject_type_v1_20260523_2120_p2_h789_fixed_w050 | p2 | 0.8750 | 0.7500 | 1.0000 | 2 | 1 | 9 | 1.0000 | 1.0000 | true |  |
| dep_hc_subject_type_v1_20260523_2230_p1_seed2026_fold0_fixed_w050 | p1 | 0.8125 | 0.8750 | 0.7500 | 6 | 4 | 2 | 0.8333 | 0.7500 | true |  |
| dep_hc_subject_type_v1_20260523_2230_p1_seed2026_fold1_fixed_w050 | p1 | 0.7500 | 0.7500 | 0.7500 | 5 | 4 | 3 | 1.0000 | 0.7500 | true |  |
| dep_hc_subject_type_v1_20260523_2230_p1_seed2026_fold2_fixed_w050 | p1 | 0.7500 | 0.7500 | 0.7500 | 5 | 4 | 3 | 0.8000 | 0.7500 | true |  |
| dep_hc_subject_type_v1_20260523_2230_p1_seed2026_fold3_fixed_w050 | p1 | 0.8125 | 0.8750 | 0.7500 | 7 | 3 | 2 | 0.8571 | 0.6667 | false | dep_like_precision |
| dep_hc_subject_type_v1_20260523_2230_p1_seed2026_fold4_fixed_w050 | p1 | 0.8750 | 0.7500 | 1.0000 | 3 | 6 | 3 | 1.0000 | 0.6667 | false | dep_like_precision |
| dep_hc_subject_type_v1_20260523_2230_p1_seed7_fold0_fixed_w050 | p1 | 0.8125 | 0.6250 | 1.0000 | 2 | 4 | 6 | 1.0000 | 0.5000 | false | dep_like_precision |
| dep_hc_subject_type_v1_20260523_2230_p1_seed7_fold1_fixed_w050 | p1 | 0.8125 | 0.6250 | 1.0000 | 4 | 5 | 3 | 1.0000 | 0.8000 | true |  |
| dep_hc_subject_type_v1_20260523_2230_p1_seed7_fold2_fixed_w050 | p1 | 0.9375 | 0.8750 | 1.0000 | 5 | 3 | 4 | 1.0000 | 1.0000 | true |  |
| dep_hc_subject_type_v1_20260523_2230_p1_seed7_fold3_fixed_w050 | p1 | 0.6250 | 0.7500 | 0.5000 | 5 | 1 | 6 | 1.0000 | 1.0000 | true |  |
| dep_hc_subject_type_v1_20260523_2230_p1_seed7_fold4_fixed_w050 | p1 | 0.7500 | 0.7500 | 0.7500 | 6 | 5 | 1 | 0.8333 | 0.6000 | false | dep_like_precision |
| dep_hc_subject_type_v1_20260523_2230_p2_h111_fixed_w050 | p2 | 0.7500 | 0.7500 | 0.7500 | 5 | 4 | 3 | 0.8000 | 0.5000 | false | dep_like_precision |
| dep_hc_subject_type_v1_20260523_2230_p2_h222_fixed_w050 | p2 | 0.6875 | 0.8750 | 0.5000 | 5 | 2 | 5 | 0.8000 | 0.5000 | false | dep_like_precision |
| dep_hc_subject_type_v1_20260523_2230_p2_h333_fixed_w050 | p2 | 0.8750 | 0.7500 | 1.0000 | 5 | 5 | 2 | 1.0000 | 0.8000 | true |  |
| dep_hc_subject_type_v1_20260523_2230_p2_h666_fixed_w050 | p2 | 0.4375 | 0.3750 | 0.5000 | 1 | 3 | 8 | 1.0000 | 0.6667 | false | subject_ba;hc_recall;dep_like_precision |

## Interpretation

- `passes_gate=false` means the classifier should remain diagnostic-only.
- Expert routing should remain offline-only until both P1 and P2 aggregate gates pass.
- Subject identifiers in this report are audit metadata only and are not model features.
