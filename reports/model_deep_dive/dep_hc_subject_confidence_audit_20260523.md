# DEP/HC Subject Confidence Audit

This report is diagnostic-only and must not be used as candidate evidence or route status.

High-confidence error threshold: `0.75`. Ambiguous p(DEP) interval: `[0.35, 0.65]`.

## Run Summary

| run | protocol | errors | high-conf errors | ambiguous | max wrong conf | mean wrong conf |
|---|---|---:|---:|---:|---:|---:|
| dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h123_w050 | p2 | 4 | 2 | 7 | 0.8385 | 0.7118 |
| dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h456_w050 | p2 | 2 | 1 | 1 | 0.7880 | 0.7431 |
| dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h999_w050 | p2 | 4 | 0 | 8 | 0.6207 | 0.5861 |
| dep_hc_subject_type_v1_20260523_2120_p1_fold0_fixed_w050 | p1 | 3 | 1 | 4 | 0.8392 | 0.6806 |
| dep_hc_subject_type_v1_20260523_2120_p1_fold1_fixed_w050 | p1 | 2 | 1 | 4 | 0.8163 | 0.6822 |
| dep_hc_subject_type_v1_20260523_2120_p1_fold2_fixed_w050 | p1 | 1 | 1 | 3 | 0.7861 | 0.7861 |
| dep_hc_subject_type_v1_20260523_2120_p1_fold3_fixed_w050 | p1 | 4 | 2 | 3 | 0.9205 | 0.7746 |
| dep_hc_subject_type_v1_20260523_2120_p1_fold4_fixed_w050 | p1 | 3 | 0 | 3 | 0.7077 | 0.6201 |
| dep_hc_subject_type_v1_20260523_2120_p2_h1024_fixed_w050 | p2 | 2 | 0 | 3 | 0.6669 | 0.6190 |
| dep_hc_subject_type_v1_20260523_2120_p2_h789_fixed_w050 | p2 | 2 | 0 | 9 | 0.6125 | 0.5897 |

## High-Confidence Errors

| run | protocol | subject | cohort | predicted | p(DEP) | confidence |
|---|---|---|---|---|---:|---:|
| dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h123_w050 | p2 | DEP1003 | DEP | HC | 0.1615 | 0.8385 |
| dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h123_w050 | p2 | HC1007 | HC | DEP | 0.7689 | 0.7689 |
| dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h456_w050 | p2 | HC1032 | HC | DEP | 0.7880 | 0.7880 |
| dep_hc_subject_type_v1_20260523_2120_p1_fold0_fixed_w050 | p1 | HC1032 | HC | DEP | 0.8392 | 0.8392 |
| dep_hc_subject_type_v1_20260523_2120_p1_fold1_fixed_w050 | p1 | HC1068 | HC | DEP | 0.8163 | 0.8163 |
| dep_hc_subject_type_v1_20260523_2120_p1_fold2_fixed_w050 | p1 | HC1026 | HC | DEP | 0.7861 | 0.7861 |
| dep_hc_subject_type_v1_20260523_2120_p1_fold3_fixed_w050 | p1 | DEP1003 | DEP | HC | 0.0795 | 0.9205 |
| dep_hc_subject_type_v1_20260523_2120_p1_fold3_fixed_w050 | p1 | DEP1015 | DEP | HC | 0.1767 | 0.8233 |
