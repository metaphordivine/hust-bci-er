# DEP/HC Diagnostic Baseline Board

This report is diagnostic-only. It does not change route status and is not candidate evidence.

## Aggregate Summary

| protocol | feature/fusion | weight source | n | mean BA | min BA | mean HC recall | mean DEP recall |
|---|---|---|---:|---:|---:|---:|---:|
| p1 | graph_connectivity |  | 1 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| p1 | traditional |  | 3 | 0.6042 | 0.5625 | 0.7083 | 0.5000 |
| p1 | traditional+time_frequency | fixed | 5 | 0.8000 | 0.6250 | 0.7500 | 0.8500 |
| p2 | graph_connectivity |  | 2 | 0.6250 | 0.4375 | 0.6250 | 0.6250 |
| p2 | traditional |  | 6 | 0.6146 | 0.4375 | 0.5208 | 0.7083 |
| p2 | traditional+time_frequency | fixed | 5 | 0.8000 | 0.6875 | 0.7000 | 0.9000 |
| p2 | traditional+time_frequency | validation_subjects | 3 | 0.7708 | 0.6875 | 0.7083 | 0.8333 |

## Baseline Runs

| run | protocol | feature/fusion | threshold | weight source | subject BA | HC recall | DEP recall | DEP/HC |
|---|---|---|---|---|---:|---:|---:|---:|
| dep_hc_region_graph_20260523_2110_p1_fold3 | p1 | graph_connectivity | fixed_0_5/fixed_0.5 |  | 0.5000 | 0.5000 | 0.5000 | 1.0000 |
| dep_hc_feature_probe_20260523_2052_p1_fold3_traditional | p1 | traditional | fixed_0_5/fixed_0.5 |  | 0.6250 | 0.7500 | 0.5000 | 0.6667 |
| dep_hc_weak_20260523_2110_p1_fold3_trad_w5 | p1 | traditional | fixed_0_5/fixed_0.5 |  | 0.5625 | 0.6250 | 0.5000 | 0.8000 |
| dep_hc_weak_20260523_2110_p1_fold3_zscore_trad | p1 | traditional | fixed_0_5/fixed_0.5 |  | 0.6250 | 0.7500 | 0.5000 | 0.6667 |
| dep_hc_subject_type_v1_20260523_2120_p1_fold0_fixed_w050 | p1 | traditional+time_frequency | fixed_0_5/fixed_0.5 | fixed | 0.8125 | 0.6250 | 1.0000 | 1.6000 |
| dep_hc_subject_type_v1_20260523_2120_p1_fold1_fixed_w050 | p1 | traditional+time_frequency | fixed_0_5/fixed_0.5 | fixed | 0.8750 | 0.7500 | 1.0000 | 1.3333 |
| dep_hc_subject_type_v1_20260523_2120_p1_fold2_fixed_w050 | p1 | traditional+time_frequency | fixed_0_5/fixed_0.5 | fixed | 0.9375 | 0.8750 | 1.0000 | 1.1429 |
| dep_hc_subject_type_v1_20260523_2120_p1_fold3_fixed_w050 | p1 | traditional+time_frequency | fixed_0_5/fixed_0.5 | fixed | 0.6250 | 0.7500 | 0.5000 | 0.6667 |
| dep_hc_subject_type_v1_20260523_2120_p1_fold4_fixed_w050 | p1 | traditional+time_frequency | fixed_0_5/fixed_0.5 | fixed | 0.7500 | 0.7500 | 0.7500 | 1.0000 |
| dep_hc_region_graph_20260523_2110_p2_h123 | p2 | graph_connectivity | fixed_0_5/fixed_0.5 |  | 0.4375 | 0.3750 | 0.5000 | 1.3333 |
| dep_hc_region_graph_20260523_2110_p2_h456 | p2 | graph_connectivity | fixed_0_5/fixed_0.5 |  | 0.8125 | 0.8750 | 0.7500 | 0.8571 |
| dep_hc_feature_probe_20260523_2052_p2_h123_traditional | p2 | traditional | fixed_0_5/fixed_0.5 |  | 0.6250 | 0.5000 | 0.7500 | 1.5000 |
| dep_hc_feature_probe_20260523_2052_p2_h999_traditional | p2 | traditional | fixed_0_5/fixed_0.5 |  | 0.6250 | 0.5000 | 0.7500 | 1.5000 |
| dep_hc_subject_type_v1_20260523_2120_p2_h1024_traditional | p2 | traditional | fixed_0_5/fixed_0.5 |  | 0.7500 | 0.7500 | 0.7500 | 1.0000 |
| dep_hc_subject_type_v1_20260523_2120_p2_h789_traditional | p2 | traditional | fixed_0_5/fixed_0.5 |  | 0.6875 | 0.6250 | 0.7500 | 1.2000 |
| dep_hc_weak_20260523_2110_p2_h123_trad_w5 | p2 | traditional | fixed_0_5/fixed_0.5 |  | 0.5625 | 0.3750 | 0.7500 | 2.0000 |
| dep_hc_weak_20260523_2110_p2_h123_zscore_trad | p2 | traditional | fixed_0_5/fixed_0.5 |  | 0.4375 | 0.3750 | 0.5000 | 1.3333 |
| dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h123_w050 | p2 | traditional+time_frequency | fixed_0_5/fixed_0.5 | fixed | 0.6875 | 0.6250 | 0.7500 | 1.2000 |
| dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h456_w050 | p2 | traditional+time_frequency | fixed_0_5/fixed_0.5 | fixed | 0.8125 | 0.8750 | 0.7500 | 0.8571 |
| dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h999_w050 | p2 | traditional+time_frequency | fixed_0_5/fixed_0.5 | fixed | 0.7500 | 0.5000 | 1.0000 | 2.0000 |
| dep_hc_fusion_20260523_184727_p2_traditional_time_frequency_h123_fixed05 | p2 | traditional+time_frequency | fixed_0_5/fixed_0.5 | validation_subjects | 0.6875 | 0.6250 | 0.7500 | 1.2000 |
| dep_hc_fusion_20260523_184727_p2_traditional_time_frequency_h456_fixed05 | p2 | traditional+time_frequency | fixed_0_5/fixed_0.5 | validation_subjects | 0.8125 | 0.8750 | 0.7500 | 0.8571 |
| dep_hc_fusion_20260523_184727_p2_traditional_time_frequency_h999_fixed05 | p2 | traditional+time_frequency | fixed_0_5/fixed_0.5 | validation_subjects | 0.8125 | 0.6250 | 1.0000 | 1.6000 |
| dep_hc_subject_type_v1_20260523_2120_p2_h1024_fixed_w050 | p2 | traditional+time_frequency | fixed_0_5/fixed_0.5 | fixed | 0.8750 | 0.7500 | 1.0000 | 1.3333 |
| dep_hc_subject_type_v1_20260523_2120_p2_h789_fixed_w050 | p2 | traditional+time_frequency | fixed_0_5/fixed_0.5 | fixed | 0.8750 | 0.7500 | 1.0000 | 1.3333 |

## Repeated Hard Subjects

| subject | cohort | errors | mean p(DEP) when wrong | runs |
|---|---|---:|---:|---|
| DEP1003 | DEP | 11 | 0.1325 | `dep_hc_feature_probe_20260523_2052_p1_fold3_traditional;dep_hc_feature_probe_20260523_2052_p2_h123_traditional;dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h123_w050;dep_hc_fusion_20260523_184727_p2_traditional_time_frequency_h123_fixed05;dep_hc_region_graph_20260523_2110_p1_fold3;dep_hc_region_graph_20260523_2110_p2_h123;dep_hc_subject_type_v1_20260523_2120_p1_fold3_fixed_w050;dep_hc_weak_20260523_2110_p1_fold3_trad_w5;dep_hc_weak_20260523_2110_p1_fold3_zscore_trad;dep_hc_weak_20260523_2110_p2_h123_trad_w5;dep_hc_weak_20260523_2110_p2_h123_zscore_trad` |
| HC1006 | HC | 10 | 0.6670 | `dep_hc_feature_probe_20260523_2052_p2_h123_traditional;dep_hc_feature_probe_20260523_2052_p2_h999_traditional;dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h123_w050;dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h999_w050;dep_hc_fusion_20260523_184727_p2_traditional_time_frequency_h123_fixed05;dep_hc_region_graph_20260523_2110_p2_h123;dep_hc_subject_type_v1_20260523_2120_p2_h789_traditional;dep_hc_weak_20260523_2110_p1_fold3_trad_w5;dep_hc_weak_20260523_2110_p2_h123_trad_w5;dep_hc_weak_20260523_2110_p2_h123_zscore_trad` |
| HC1038 | HC | 9 | 0.7047 | `dep_hc_feature_probe_20260523_2052_p2_h123_traditional;dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h123_w050;dep_hc_fusion_20260523_184727_p2_traditional_time_frequency_h123_fixed05;dep_hc_region_graph_20260523_2110_p2_h123;dep_hc_subject_type_v1_20260523_2120_p1_fold0_fixed_w050;dep_hc_subject_type_v1_20260523_2120_p2_h1024_fixed_w050;dep_hc_subject_type_v1_20260523_2120_p2_h1024_traditional;dep_hc_weak_20260523_2110_p2_h123_trad_w5;dep_hc_weak_20260523_2110_p2_h123_zscore_trad` |
| HC1005 | HC | 7 | 0.6539 | `dep_hc_feature_probe_20260523_2052_p1_fold3_traditional;dep_hc_region_graph_20260523_2110_p1_fold3;dep_hc_subject_type_v1_20260523_2120_p1_fold3_fixed_w050;dep_hc_subject_type_v1_20260523_2120_p2_h789_fixed_w050;dep_hc_subject_type_v1_20260523_2120_p2_h789_traditional;dep_hc_weak_20260523_2110_p1_fold3_trad_w5;dep_hc_weak_20260523_2110_p1_fold3_zscore_trad` |
| HC1007 | HC | 6 | 0.7568 | `dep_hc_feature_probe_20260523_2052_p2_h123_traditional;dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h123_w050;dep_hc_fusion_20260523_184727_p2_traditional_time_frequency_h123_fixed05;dep_hc_region_graph_20260523_2110_p2_h123;dep_hc_weak_20260523_2110_p2_h123_trad_w5;dep_hc_weak_20260523_2110_p2_h123_zscore_trad` |
| HC1042 | HC | 5 | 0.6150 | `dep_hc_feature_probe_20260523_2052_p2_h999_traditional;dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h999_w050;dep_hc_fusion_20260523_184727_p2_traditional_time_frequency_h999_fixed05;dep_hc_region_graph_20260523_2110_p2_h123;dep_hc_subject_type_v1_20260523_2120_p1_fold1_fixed_w050` |
| DEP1015 | DEP | 4 | 0.1408 | `dep_hc_feature_probe_20260523_2052_p1_fold3_traditional;dep_hc_subject_type_v1_20260523_2120_p1_fold3_fixed_w050;dep_hc_weak_20260523_2110_p1_fold3_trad_w5;dep_hc_weak_20260523_2110_p1_fold3_zscore_trad` |
| HC1010 | HC | 4 | 0.8245 | `dep_hc_feature_probe_20260523_2052_p1_fold3_traditional;dep_hc_subject_type_v1_20260523_2120_p1_fold3_fixed_w050;dep_hc_weak_20260523_2110_p1_fold3_trad_w5;dep_hc_weak_20260523_2110_p1_fold3_zscore_trad` |
| HC1016 | HC | 4 | 0.6670 | `dep_hc_feature_probe_20260523_2052_p2_h123_traditional;dep_hc_region_graph_20260523_2110_p2_h123;dep_hc_weak_20260523_2110_p2_h123_trad_w5;dep_hc_weak_20260523_2110_p2_h123_zscore_trad` |
| HC1068 | HC | 4 | 0.6460 | `dep_hc_feature_probe_20260523_2052_p2_h999_traditional;dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h999_w050;dep_hc_fusion_20260523_184727_p2_traditional_time_frequency_h999_fixed05;dep_hc_subject_type_v1_20260523_2120_p1_fold1_fixed_w050` |
| HC1011 | HC | 3 | 0.6288 | `dep_hc_feature_probe_20260523_2052_p2_h999_traditional;dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h999_w050;dep_hc_fusion_20260523_184727_p2_traditional_time_frequency_h999_fixed05` |
| DEP1022 | DEP | 3 | 0.3511 | `dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h456_w050;dep_hc_fusion_20260523_184727_p2_traditional_time_frequency_h456_fixed05;dep_hc_subject_type_v1_20260523_2120_p1_fold4_fixed_w050` |
| HC1032 | HC | 3 | 0.8051 | `dep_hc_fixed_weight_20260523_2040_p2_trad_tf_h456_w050;dep_hc_fusion_20260523_184727_p2_traditional_time_frequency_h456_fixed05;dep_hc_subject_type_v1_20260523_2120_p1_fold0_fixed_w050` |
| HC1039 | HC | 3 | 0.7278 | `dep_hc_subject_type_v1_20260523_2120_p1_fold4_fixed_w050;dep_hc_subject_type_v1_20260523_2120_p2_h1024_fixed_w050;dep_hc_subject_type_v1_20260523_2120_p2_h1024_traditional` |
| DEP1018 | DEP | 2 | 0.1524 | `dep_hc_region_graph_20260523_2110_p1_fold3;dep_hc_region_graph_20260523_2110_p2_h123` |
| DEP1010 | DEP | 2 | 0.4557 | `dep_hc_subject_type_v1_20260523_2120_p2_h1024_traditional;dep_hc_subject_type_v1_20260523_2120_p2_h789_traditional` |
| HC1051 | HC | 2 | 0.6708 | `dep_hc_subject_type_v1_20260523_2120_p2_h789_fixed_w050;dep_hc_subject_type_v1_20260523_2120_p2_h789_traditional` |
| HC1019 | HC | 2 | 0.5380 | `dep_hc_weak_20260523_2110_p2_h123_trad_w5;dep_hc_weak_20260523_2110_p2_h123_zscore_trad` |
| DEP1030 | DEP | 1 | 0.4504 | `dep_hc_feature_probe_20260523_2052_p2_h999_traditional` |
| HC1013 | HC | 1 | 0.6035 | `dep_hc_region_graph_20260523_2110_p1_fold3` |

## Interpretation Guardrails

- Use this board to compare diagnostic baselines only.
- Report validation-selected fusion and fixed 0.5/0.5 fusion side by side.
- Subject identifiers shown here are audit/report metadata only, never model features.
- Keep DEP/HC classifier work isolated from emotion Top-4 route evidence until it passes stronger protocol gates.
