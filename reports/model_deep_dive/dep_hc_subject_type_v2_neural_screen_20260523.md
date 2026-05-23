# DEP/HC Diagnostic Baseline Board

This report is diagnostic-only. It does not change route status and is not candidate evidence.

## Aggregate Summary

| protocol | feature/fusion | weight source | n | mean BA | min BA | mean HC recall | mean DEP recall |
|---|---|---|---:|---:|---:|---:|---:|
| p1 | neural:deformer_lite |  | 1 | 0.6250 | 0.6250 | 0.7500 | 0.5000 |
| p2 | neural:conformer_lite |  | 3 | 0.5417 | 0.5000 | 0.8333 | 0.2500 |
| p2 | neural:deformer_lite |  | 8 | 0.5781 | 0.4375 | 0.8125 | 0.3438 |
| p2 | neural:dgcnn |  | 2 | 0.3750 | 0.2500 | 0.6250 | 0.1250 |
| p2 | neural:dual_graph_conformer |  | 2 | 0.5312 | 0.4375 | 0.9375 | 0.1250 |
| p2 | neural:fbstcnet |  | 6 | 0.5208 | 0.3750 | 0.7083 | 0.3333 |
| p2 | neural:srfnet |  | 1 | 0.5000 | 0.5000 | 1.0000 | 0.0000 |
| p2 | neural:tsception |  | 1 | 0.4375 | 0.4375 | 0.3750 | 0.5000 |

## Baseline Runs

| run | protocol | feature/fusion | threshold | weight source | subject BA | HC recall | DEP recall | DEP/HC |
|---|---|---|---|---|---:|---:|---:|---:|
| dep_hc_subject_type_v2_cuda_p1_fold3_deformer_attn_e8_ba | p1 | neural:deformer_lite | balanced_accuracy/validation_subjects |  | 0.6250 | 0.7500 | 0.5000 | 0.6667 |
| dep_hc_subject_type_v2_cuda_p2_h123_conformer_e8_ba | p2 | neural:conformer_lite | balanced_accuracy/validation_subjects |  | 0.5000 | 1.0000 | 0.0000 | 0.0000 |
| dep_hc_subject_type_v2_screen_p2_h666_conformer_e8_ba | p2 | neural:conformer_lite | balanced_accuracy/validation_subjects |  | 0.5625 | 0.6250 | 0.5000 | 0.8000 |
| dep_hc_subject_type_v2_smoke_p2_h666_conformer_e2 | p2 | neural:conformer_lite | fixed_0_5/fixed_0.5 |  | 0.5625 | 0.8750 | 0.2500 | 0.2857 |
| dep_hc_subject_type_v2_cuda_p2_h123_deformer_attn_e16_ba | p2 | neural:deformer_lite | balanced_accuracy/validation_subjects |  | 0.5625 | 0.8750 | 0.2500 | 0.2857 |
| dep_hc_subject_type_v2_cuda_p2_h123_deformer_attn_e8_ba | p2 | neural:deformer_lite | balanced_accuracy/validation_subjects |  | 0.6250 | 1.0000 | 0.2500 | 0.2500 |
| dep_hc_subject_type_v2_cuda_p2_h123_deformer_attn_e8_fixed | p2 | neural:deformer_lite | fixed_0_5/fixed_0.5 |  | 0.5625 | 0.8750 | 0.2500 | 0.2857 |
| dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e16_ba | p2 | neural:deformer_lite | balanced_accuracy/validation_subjects |  | 0.5625 | 0.8750 | 0.2500 | 0.2857 |
| dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e8_ba | p2 | neural:deformer_lite | balanced_accuracy/validation_subjects |  | 0.6250 | 0.5000 | 0.7500 | 1.5000 |
| dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e8_depfloor | p2 | neural:deformer_lite | dep_recall_floor_0p8_hc/validation_subjects |  | 0.4375 | 0.6250 | 0.2500 | 0.4000 |
| dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e8_fixed | p2 | neural:deformer_lite | fixed_0_5/fixed_0.5 |  | 0.5000 | 1.0000 | 0.0000 | 0.0000 |
| dep_hc_subject_type_v2_cuda_p2_h999_deformer_attn_e8_ba | p2 | neural:deformer_lite | balanced_accuracy/validation_subjects |  | 0.7500 | 0.7500 | 0.7500 | 1.0000 |
| dep_hc_subject_type_v2_cuda_p2_h123_dgcnn_e8_ba | p2 | neural:dgcnn | balanced_accuracy/validation_subjects |  | 0.5000 | 1.0000 | 0.0000 | 0.0000 |
| dep_hc_subject_type_v2_cuda_p2_h666_dgcnn_e8_ba | p2 | neural:dgcnn | balanced_accuracy/validation_subjects |  | 0.2500 | 0.2500 | 0.2500 | 1.0000 |
| dep_hc_subject_type_v2_cuda_p2_h123_dual_graph_conformer_e8_ba | p2 | neural:dual_graph_conformer | balanced_accuracy/validation_subjects |  | 0.6250 | 1.0000 | 0.2500 | 0.2500 |
| dep_hc_subject_type_v2_cuda_p2_h666_dual_graph_conformer_e8_ba | p2 | neural:dual_graph_conformer | balanced_accuracy/validation_subjects |  | 0.4375 | 0.8750 | 0.0000 | 0.0000 |
| dep_hc_subject_type_v2_cuda_p2_h123_fbstcnet_e8_ba | p2 | neural:fbstcnet | balanced_accuracy/validation_subjects |  | 0.5000 | 0.7500 | 0.2500 | 0.3333 |
| dep_hc_subject_type_v2_cuda_p2_h123_fbstcnet_route_e8_ba | p2 | neural:fbstcnet | balanced_accuracy/validation_subjects |  | 0.3750 | 0.7500 | 0.0000 | 0.0000 |
| dep_hc_subject_type_v2_cuda_p2_h666_fbstcnet_route_e8_ba | p2 | neural:fbstcnet | balanced_accuracy/validation_subjects |  | 0.6875 | 0.6250 | 0.7500 | 1.2000 |
| dep_hc_subject_type_v2_cuda_p2_h999_fbstcnet_route_e8_ba | p2 | neural:fbstcnet | balanced_accuracy/validation_subjects |  | 0.5000 | 0.7500 | 0.2500 | 0.3333 |
| dep_hc_subject_type_v2_screen_p2_h666_fbstcnet_e8_ba | p2 | neural:fbstcnet | balanced_accuracy/validation_subjects |  | 0.5625 | 0.6250 | 0.5000 | 0.8000 |
| dep_hc_subject_type_v2_smoke_p2_h666_fbstcnet_e2 | p2 | neural:fbstcnet | fixed_0_5/fixed_0.5 |  | 0.5000 | 0.7500 | 0.2500 | 0.3333 |
| dep_hc_subject_type_v2_smoke_p2_h666_srfnet_e2 | p2 | neural:srfnet | fixed_0_5/fixed_0.5 |  | 0.5000 | 1.0000 | 0.0000 | 0.0000 |
| dep_hc_subject_type_v2_smoke_p2_h666_tsception_e2 | p2 | neural:tsception | fixed_0_5/fixed_0.5 |  | 0.4375 | 0.3750 | 0.5000 | 1.3333 |

## Repeated Hard Subjects

| subject | cohort | errors | mean p(DEP) when wrong | runs |
|---|---|---:|---:|---|
| DEP1018 | DEP | 21 | 0.1272 | `dep_hc_subject_type_v2_cuda_p2_h123_conformer_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_deformer_attn_e16_ba;dep_hc_subject_type_v2_cuda_p2_h123_deformer_attn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_deformer_attn_e8_fixed;dep_hc_subject_type_v2_cuda_p2_h123_dgcnn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_dual_graph_conformer_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_fbstcnet_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_fbstcnet_route_e8_ba;dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e16_ba;dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e8_depfloor;dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e8_fixed;dep_hc_subject_type_v2_cuda_p2_h666_dgcnn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h666_dual_graph_conformer_e8_ba;dep_hc_subject_type_v2_cuda_p2_h666_fbstcnet_route_e8_ba;dep_hc_subject_type_v2_screen_p2_h666_conformer_e8_ba;dep_hc_subject_type_v2_screen_p2_h666_fbstcnet_e8_ba;dep_hc_subject_type_v2_smoke_p2_h666_conformer_e2;dep_hc_subject_type_v2_smoke_p2_h666_fbstcnet_e2;dep_hc_subject_type_v2_smoke_p2_h666_srfnet_e2;dep_hc_subject_type_v2_smoke_p2_h666_tsception_e2` |
| DEP1028 | DEP | 11 | 0.1375 | `dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e16_ba;dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e8_depfloor;dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e8_fixed;dep_hc_subject_type_v2_cuda_p2_h666_dgcnn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h666_dual_graph_conformer_e8_ba;dep_hc_subject_type_v2_cuda_p2_h999_deformer_attn_e8_ba;dep_hc_subject_type_v2_screen_p2_h666_conformer_e8_ba;dep_hc_subject_type_v2_smoke_p2_h666_conformer_e2;dep_hc_subject_type_v2_smoke_p2_h666_fbstcnet_e2;dep_hc_subject_type_v2_smoke_p2_h666_srfnet_e2;dep_hc_subject_type_v2_smoke_p2_h666_tsception_e2` |
| HC1011 | HC | 10 | 0.5180 | `dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e8_depfloor;dep_hc_subject_type_v2_cuda_p2_h666_dgcnn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h666_fbstcnet_route_e8_ba;dep_hc_subject_type_v2_cuda_p2_h999_deformer_attn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h999_fbstcnet_route_e8_ba;dep_hc_subject_type_v2_screen_p2_h666_conformer_e8_ba;dep_hc_subject_type_v2_screen_p2_h666_fbstcnet_e8_ba;dep_hc_subject_type_v2_smoke_p2_h666_fbstcnet_e2;dep_hc_subject_type_v2_smoke_p2_h666_tsception_e2` |
| DEP1003 | DEP | 9 | 0.1525 | `dep_hc_subject_type_v2_cuda_p1_fold3_deformer_attn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_conformer_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_deformer_attn_e16_ba;dep_hc_subject_type_v2_cuda_p2_h123_deformer_attn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_deformer_attn_e8_fixed;dep_hc_subject_type_v2_cuda_p2_h123_dgcnn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_dual_graph_conformer_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_fbstcnet_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_fbstcnet_route_e8_ba` |
| DEP1023 | DEP | 8 | 0.2190 | `dep_hc_subject_type_v2_cuda_p2_h123_conformer_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_deformer_attn_e16_ba;dep_hc_subject_type_v2_cuda_p2_h123_deformer_attn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_deformer_attn_e8_fixed;dep_hc_subject_type_v2_cuda_p2_h123_dgcnn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_dual_graph_conformer_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_fbstcnet_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_fbstcnet_route_e8_ba` |
| HC1068 | HC | 8 | 0.3976 | `dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e16_ba;dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e8_depfloor;dep_hc_subject_type_v2_cuda_p2_h666_dgcnn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h666_dual_graph_conformer_e8_ba;dep_hc_subject_type_v2_screen_p2_h666_conformer_e8_ba;dep_hc_subject_type_v2_smoke_p2_h666_conformer_e2;dep_hc_subject_type_v2_smoke_p2_h666_tsception_e2` |
| DEP1033 | DEP | 7 | 0.1750 | `dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e16_ba;dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e8_depfloor;dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e8_fixed;dep_hc_subject_type_v2_cuda_p2_h666_dual_graph_conformer_e8_ba;dep_hc_subject_type_v2_screen_p2_h666_fbstcnet_e8_ba;dep_hc_subject_type_v2_smoke_p2_h666_fbstcnet_e2;dep_hc_subject_type_v2_smoke_p2_h666_srfnet_e2` |
| HC1019 | HC | 6 | 0.3857 | `dep_hc_subject_type_v2_cuda_p2_h123_fbstcnet_route_e8_ba;dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h666_dgcnn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h666_fbstcnet_route_e8_ba;dep_hc_subject_type_v2_screen_p2_h666_conformer_e8_ba;dep_hc_subject_type_v2_smoke_p2_h666_tsception_e2` |
| HC1025 | HC | 6 | 0.3688 | `dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e8_depfloor;dep_hc_subject_type_v2_cuda_p2_h666_dgcnn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h666_fbstcnet_route_e8_ba;dep_hc_subject_type_v2_screen_p2_h666_fbstcnet_e8_ba;dep_hc_subject_type_v2_smoke_p2_h666_fbstcnet_e2` |
| DEP1025 | DEP | 5 | 0.1425 | `dep_hc_subject_type_v2_cuda_p2_h666_deformer_attn_e8_fixed;dep_hc_subject_type_v2_cuda_p2_h666_dgcnn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h666_dual_graph_conformer_e8_ba;dep_hc_subject_type_v2_smoke_p2_h666_conformer_e2;dep_hc_subject_type_v2_smoke_p2_h666_srfnet_e2` |
| DEP1024 | DEP | 4 | 0.5393 | `dep_hc_subject_type_v2_cuda_p2_h123_conformer_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_dgcnn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_fbstcnet_route_e8_ba;dep_hc_subject_type_v2_cuda_p2_h999_fbstcnet_route_e8_ba` |
| HC1007 | HC | 4 | 0.6170 | `dep_hc_subject_type_v2_cuda_p2_h123_deformer_attn_e16_ba;dep_hc_subject_type_v2_cuda_p2_h123_deformer_attn_e8_fixed;dep_hc_subject_type_v2_cuda_p2_h123_fbstcnet_e8_ba;dep_hc_subject_type_v2_cuda_p2_h123_fbstcnet_route_e8_ba` |
| HC1005 | HC | 2 | 0.0772 | `dep_hc_subject_type_v2_cuda_p1_fold3_deformer_attn_e8_ba;dep_hc_subject_type_v2_cuda_p2_h666_dgcnn_e8_ba` |
| HC1006 | HC | 2 | 0.5981 | `dep_hc_subject_type_v2_cuda_p2_h123_fbstcnet_e8_ba;dep_hc_subject_type_v2_cuda_p2_h999_fbstcnet_route_e8_ba` |
| HC1028 | HC | 2 | 0.3405 | `dep_hc_subject_type_v2_cuda_p2_h666_dgcnn_e8_ba;dep_hc_subject_type_v2_smoke_p2_h666_tsception_e2` |
| HC1051 | HC | 2 | 0.4766 | `dep_hc_subject_type_v2_screen_p2_h666_fbstcnet_e8_ba;dep_hc_subject_type_v2_smoke_p2_h666_tsception_e2` |
| DEP1015 | DEP | 1 | 0.0003 | `dep_hc_subject_type_v2_cuda_p1_fold3_deformer_attn_e8_ba` |
| HC1013 | HC | 1 | 0.0015 | `dep_hc_subject_type_v2_cuda_p1_fold3_deformer_attn_e8_ba` |
| HC1042 | HC | 1 | 0.7460 | `dep_hc_subject_type_v2_cuda_p2_h999_deformer_attn_e8_ba` |
| DEP1027 | DEP | 1 | 0.0180 | `dep_hc_subject_type_v2_cuda_p2_h999_fbstcnet_route_e8_ba` |

## Interpretation Guardrails

- Use this board to compare diagnostic baselines only.
- Report validation-selected fusion and fixed 0.5/0.5 fusion side by side.
- Subject identifiers shown here are audit/report metadata only, never model features.
- Keep DEP/HC classifier work isolated from emotion Top-4 route evidence until it passes stronger protocol gates.
