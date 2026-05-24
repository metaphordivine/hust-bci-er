# All Evaluation Score Table 2026-05-24

Scope: cleaned ledger of score-bearing evaluation artifacts that are already represented in repository reports. This report does not scan untracked raw output directories, prediction tables, checkpoints, or public/private labels.

- Detail CSV: `reports/model_deep_dive/all_evaluation_score_table_20260524.csv`
- Route summary CSV: `reports/model_deep_dive/all_evaluation_route_summary_20260524.csv`
- Remote P1/P2/P3 table: `reports/model_deep_dive/remote_protocol_route_score_table_20260524.csv`
- Remote protocol ledgers scanned: 2
- Route status is unchanged; this is score organization, not promotion evidence.

## Discovery

- Tracked report files scanned: 270
- Tracked outputs/scratch files scanned: 0
- Route-level rows in clean summary: 93
- Detail score records: 1336

| source_family | records |
|---|---|
| candidate_audit_json | 48 |
| lightweight_protocol_board | 21 |
| model_deep_dive_csv | 85 |
| model_deep_dive_markdown_table | 474 |
| remote_protocol_ledger_run | 505 |
| remote_protocol_route_table | 140 |
| route_summary | 61 |
| route_summary_protocol_metric | 2 |

## Protocol Record Counts

| protocol | records |
|---|---|
| P1 | 457 |
| P2 | 341 |
| P3 | 295 |
| score_fusion margin_adaptive_query_context | 2 |
| score_fusion query_context | 3 |
| score_fusion sparse query_context | 1 |
| unspecified | 237 |

## Remote Protocol P1 5-Seed Leaders

| route | P1 seeds | P1 5seed jobs | P1 5seed mean | P2 best mean | P3 best mean |
|---|---|---|---|---|---|
| `fixed_crop_car_fbstcnet` | 42,123,456,789,1024 | 25/25 | 0.7146 | 0.6756 | 0.7250 |
| `fixed_crop_whitening_eps3e4_fbstcnet_m_conn_light` | 42,123,456,789,1024 | 25/25 | 0.7102 | 0.6548 | 0.7128 |
| `fixed_crop_whitening_eps1e3_fbstcnet_m_conn_light` | 42,123,456,789,1024 | 25/25 | 0.7099 | 0.6518 | 0.7127 |
| `fixed_crop_whitening_eps3e4_fbstcnet` | 42,123,456,789,1024 | 25/25 | 0.7097 | 0.6607 | 0.7121 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet_m_conn_light` | 42,123,456,789,1024 | 25/25 | 0.7086 | 0.6577 | 0.7149 |
| `fixed_crop_whitening_eps1e3_fbstcnet` | 42,123,456,789,1024 | 25/25 | 0.7085 | 0.6577 | 0.7122 |
| `fixed_crop_ea_whitening_eps3e4_fbstcnet_m_conn_light` | 42,123,456,789,1024 | 25/25 | 0.7081 | 0.6518 | 0.7116 |
| `fixed_crop_ea_whitening_eps3e4_fbstcnet` | 42,123,456,789,1024 | 25/25 | 0.7080 | 0.6518 | 0.7085 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet` | 42,123,456,789,1024 | 25/25 | 0.7075 | 0.6548 | 0.7082 |
| `fixed_crop_whitening_eps3e4_tri_context_gate` | 42,123,456,789,1024 | 25/25 | 0.7056 | 0.6815 |  |
| `fixed_crop_ea_fbstcnet_c_only` | 42,123,456,789,1024 | 25/25 | 0.7014 | 0.6696 | 0.7124 |
| `fixed_crop_ea_whitening_eps3e4_fbstcnet_c_only` | 42,123,456,789,1024 | 25/25 | 0.7005 | 0.6429 | 0.7018 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet_c_only` | 42,123,456,789,1024 | 25/25 | 0.7000 | 0.6488 | 0.7031 |
| `fixed_crop_ea_fbstcnet_m_conn_light` | 42,123,456,789,1024 | 25/25 | 0.6992 | 0.6845 | 0.7104 |
| `fixed_crop_ea_fbstcnet_m_power_light` | 42,123,456,789,1024 | 25/25 | 0.6942 | 0.6845 | 0.7035 |

## Remote Protocol P2 7-Job Leaders

| route | P2 complete runs | P2 best mean | P2 7 jobs | P1 5seed mean | P3 best mean |
|---|---|---|---|---|---|
| `sliding_ea_dgcnn_coral_cohort_w6_s1` | 1 | 0.7054 | crop1=0.6875; crop2=0.7083; crop3=0.7500; crop4=0.7292; crop5=0.7917; random=0.7083; worst=0.5625 | 0.6319 | 0.6345 |
| `sliding_ea_dgcnn_w6_s1` | 1 | 0.6875 | crop1=0.7083; crop2=0.6875; crop3=0.7083; crop4=0.7292; crop5=0.7292; random=0.7083; worst=0.5417 | 0.6316 | 0.6282 |
| `fixed_crop_ea_dgcnn` | 1 | 0.6845 | crop1=0.7292; crop2=0.6667; crop3=0.7500; crop4=0.7500; crop5=0.7708; random=0.7500; worst=0.3750 | 0.6932 | 0.7016 |
| `fixed_crop_ea_fbstcnet_m_conn_light` | 1 | 0.6845 | crop1=0.7083; crop2=0.6875; crop3=0.7292; crop4=0.7292; crop5=0.7708; random=0.7708; worst=0.3958 | 0.6992 | 0.7104 |
| `fixed_crop_ea_fbstcnet_m_power_light` | 1 | 0.6845 | crop1=0.6875; crop2=0.7292; crop3=0.7708; crop4=0.7292; crop5=0.7500; random=0.7708; worst=0.3542 | 0.6942 | 0.7035 |
| `fixed_crop_whitening_eps3e4_tri_context_gate` | 1 | 0.6815 | crop1=0.7292; crop2=0.7083; crop3=0.7917; crop4=0.7500; crop5=0.7500; random=0.7500; worst=0.2917 | 0.7056 |  |
| `fixed_crop_ea_fbstcnet_p_only` | 1 | 0.6786 | crop1=0.6667; crop2=0.7292; crop3=0.6875; crop4=0.7500; crop5=0.7500; random=0.8125; worst=0.3542 | 0.6770 | 0.6938 |
| `fixed_crop_ea_fbstcnet_riem_guided_gate` | 1 | 0.6786 | crop1=0.6875; crop2=0.7292; crop3=0.7292; crop4=0.7500; crop5=0.7292; random=0.7500; worst=0.3750 | 0.6883 | 0.7043 |
| `fixed_crop_car_fbstcnet` | 1 | 0.6756 | crop1=0.7083; crop2=0.7292; crop3=0.7708; crop4=0.7083; crop5=0.7083; random=0.7500; worst=0.3542 | 0.7146 | 0.7250 |
| `sliding_ea_fbstcnet_m_conn_w8_s0p5` | 1 | 0.6726 | crop1=0.6875; crop2=0.6875; crop3=0.6875; crop4=0.6667; crop5=0.6667; random=0.7083; worst=0.6042 | 0.6448 |  |
| `fixed_crop_ea_fbstcnet_c_only` | 1 | 0.6696 | crop1=0.6667; crop2=0.7083; crop3=0.6875; crop4=0.7083; crop5=0.7708; random=0.7292; worst=0.4167 | 0.7014 | 0.7124 |
| `sliding_ea_dgcnn_dann_cohort_w6_s1` | 1 | 0.6667 | crop1=0.7083; crop2=0.6875; crop3=0.7083; crop4=0.6667; crop5=0.6667; random=0.7083; worst=0.5208 | 0.6407 | 0.6474 |
| `sliding_ea_dgcnn_masked_consistency_w6_s1` | 1 | 0.6637 | crop1=0.6667; crop2=0.6875; crop3=0.6667; crop4=0.6875; crop5=0.7083; random=0.7083; worst=0.5208 | 0.6408 | 0.6327 |
| `fixed_crop_whitening_eps3e4_fbstcnet` | 1 | 0.6607 | crop1=0.7083; crop2=0.7083; crop3=0.7500; crop4=0.7708; crop5=0.6875; random=0.7500; worst=0.2500 | 0.7097 | 0.7121 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet_m_conn_light` | 1 | 0.6577 | crop1=0.7500; crop2=0.6875; crop3=0.7500; crop4=0.6875; crop5=0.7500; random=0.6875; worst=0.2917 | 0.7086 | 0.7149 |

## Remote Protocol P3 5-Outer Leaders

| route | P3 complete runs | P3 best mean | P3 outer jobs | P1 5seed mean | P2 best mean |
|---|---|---|---|---|---|
| `fixed_crop_car_fbstcnet` | 1 | 0.7250 | outer0/seed42=0.6947; outer1/seed43=0.7140; outer2/seed44=0.7385; outer3/seed45=0.6860; outer4/seed46=0.7917 | 0.7146 | 0.6756 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet_m_conn_light` | 1 | 0.7149 | outer0/seed42=0.7147; outer1/seed43=0.6882; outer2/seed44=0.7127; outer3/seed45=0.6894; outer4/seed46=0.7698 | 0.7086 | 0.6577 |
| `fixed_crop_whitening_eps3e4_fbstcnet_m_conn_light` | 1 | 0.7128 | outer0/seed42=0.7053; outer1/seed43=0.6764; outer2/seed44=0.7145; outer3/seed45=0.6957; outer4/seed46=0.7722 | 0.7102 | 0.6548 |
| `fixed_crop_whitening_eps1e3_fbstcnet_m_conn_light` | 1 | 0.7127 | outer0/seed42=0.7049; outer1/seed43=0.6771; outer2/seed44=0.7146; outer3/seed45=0.6980; outer4/seed46=0.7691 | 0.7099 | 0.6518 |
| `fixed_crop_ea_fbstcnet_c_only` | 1 | 0.7124 | outer0/seed42=0.7026; outer1/seed43=0.6816; outer2/seed44=0.7251; outer3/seed45=0.6864; outer4/seed46=0.7665 | 0.7014 | 0.6696 |
| `fixed_crop_whitening_eps1e3_fbstcnet` | 1 | 0.7122 | outer0/seed42=0.7123; outer1/seed43=0.6711; outer2/seed44=0.7110; outer3/seed45=0.6968; outer4/seed46=0.7698 | 0.7085 | 0.6577 |
| `fixed_crop_whitening_eps3e4_fbstcnet` | 1 | 0.7121 | outer0/seed42=0.7129; outer1/seed43=0.6710; outer2/seed44=0.7103; outer3/seed45=0.6950; outer4/seed46=0.7713 | 0.7097 | 0.6607 |
| `fixed_crop_ea_whitening_eps3e4_fbstcnet_m_conn_light` | 1 | 0.7116 | outer0/seed42=0.7125; outer1/seed43=0.6882; outer2/seed44=0.7128; outer3/seed45=0.6901; outer4/seed46=0.7544 | 0.7081 | 0.6518 |
| `fixed_crop_ea_fbstcnet_m_conn_light` | 1 | 0.7104 | outer0/seed42=0.6915; outer1/seed43=0.7151; outer2/seed44=0.7016; outer3/seed45=0.6783; outer4/seed46=0.7657 | 0.6992 | 0.6845 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet_m_power_light` | 1 | 0.7103 | outer0/seed42=0.7178; outer1/seed43=0.6659; outer2/seed44=0.7157; outer3/seed45=0.6901; outer4/seed46=0.7621 |  | 0.6548 |
| `fixed_crop_ea_whitening_eps3e4_fbstcnet` | 1 | 0.7085 | outer0/seed42=0.7138; outer1/seed43=0.6837; outer2/seed44=0.7104; outer3/seed45=0.6855; outer4/seed46=0.7489 | 0.7080 | 0.6518 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet` | 1 | 0.7082 | outer0/seed42=0.7144; outer1/seed43=0.6829; outer2/seed44=0.7107; outer3/seed45=0.6864; outer4/seed46=0.7465 | 0.7075 | 0.6548 |
| `fixed_crop_ea_car_fbstcnet` | 1 | 0.7071 | outer0/seed42=0.7010; outer1/seed43=0.7030; outer2/seed44=0.7034; outer3/seed45=0.6599; outer4/seed46=0.7682 |  |  |
| `fixed_crop_ea_fbstcnet_riem_guided_gate` | 1 | 0.7043 | outer0/seed42=0.6727; outer1/seed43=0.7119; outer2/seed44=0.6969; outer3/seed45=0.6823; outer4/seed46=0.7576 | 0.6883 | 0.6786 |
| `fixed_crop_ea_fbstcnet_m_power_light` | 1 | 0.7035 | outer0/seed42=0.6887; outer1/seed43=0.7028; outer2/seed44=0.7098; outer3/seed45=0.6739; outer4/seed46=0.7425 | 0.6942 | 0.6845 |

## Route Summary Primary Metric Leaders

| route | status | audit | protocol | seeds | jobs | summary mean | audit json metric |
|---|---|---|---|---|---|---|---|
| `car_fbstcnet_as_query_noea_eps3e4_m_conn_srfnet_whitening_eps3e4_conformer_context_fusion` | IDEA | PASS_P1_P2_FIXED_CROPS_AND_P3 | score_fusion query_context | component manifests |  | 0.7366 |  |
| `car_fbstcnet_as_query_srfnet_reference_whitening_eps3e4_conformer_context_fusion` | IDEA | PASS_P1_P2_FIXED_CROPS_AND_P3 | score_fusion query_context | component manifests |  | 0.7364 |  |
| `car_fbstcnet_sparse_noea_m_conn_srfnet_whitening_conformer_context_fusion` | IDEA | PASS_P1_AND_P2_FIXED_CROPS | score_fusion sparse query_context | component manifests |  | 0.7310 |  |
| `car_fbstcnet_as_query_srfnet_whitening_eps3e4_conformer_context_fusion` | IDEA | PASS_P1_P2_FIXED_CROPS_AND_P3 | score_fusion query_context | protocol default unless specified by component manifests |  | 0.7307 |  |
| `car_fbstcnet_boundary_repair_noea_m_conn_srfnet_whitening_conformer_context_fusion` | IDEA | PASS_P1_AND_P2_FIXED_CROPS | score_fusion margin_adaptive_query_context | component manifests |  | 0.7306 |  |
| `car_fbstcnet_margin_adaptive_srfnet_whitening_eps3e4_conformer_context_fusion` | IDEA | PASS_P1_P2_FIXED_CROPS_AND_P3 | score_fusion margin_adaptive_query_context | protocol default unless specified by component manifests |  | 0.7267 |  |
| `fixed_crop_car_fbstcnet` | IDEA | PASS | P1 | 42 |  | 0.7065 | 0.7065 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet` | IDEA | PASS | P1 | 42 |  | 0.7036 | 0.7036 |
| `fixed_crop_ea_fbstcnet_c_only` | IDEA | PASS | P1 | 42 |  | 0.7021 | 0.7021 |
| `fbstcnet_srfnet_whitening_eps1e3_average` | IDEA | PASS | P1 | 42 |  | 0.6994 | 0.6994 |
| `dgcnn_adaptation_fbstcnet_srfnet_conformer_score_average` | IDEA | PASS | P1 | 42 |  | 0.6985 |  |
| `fixed_crop_fbstcnet_zscore_only` | IDEA | PASS | P1 | 42 |  | 0.6958 | 0.6958 |
| `fixed_crop_ea_fbstcnet_bands_4_40` | IDEA | PASS | P1 | 42 |  | 0.6945 | 0.6945 |
| `fixed_crop_whitening_eps1e3_fbstcnet` | IDEA | PASS | P1 | 42 |  | 0.6945 | 0.6945 |
| `fbstcnet_as_query_srfnet_context_fusion` | IDEA | PASS | P1 | 42 |  | 0.6934 | 0.6934 |
| `fixed_crop_ea_dgcnn` | IDEA | PENDING_REAUDIT | P1 | 42, 123, 456, 789, 1024 | 25 | 0.6932 |  |
| `fbstcnet_srfnet_score_average` | IDEA | PASS | P1 | 42 |  | 0.6931 | 0.6931 |
| `fbstcnet_srfnet_conformer_score_average` | IDEA | PASS | P1 | 42 |  | 0.6913 | 0.6913 |
| `fixed_crop_ea_fbstcnet_riem_guided_gate` | IDEA | PASS | P1 | 42, 123, 456, 789, 1024 | P1 25/25; P2 7/7 prediction jobs plus 1 train artifact job; P3 5/5 final jobs plus 15 inner selection jobs | 0.6883 |  |
| `fixed_crop_ea_fbstcnet_m_conn_light` | IDEA | PASS | P1 | 42 |  | 0.6876 | 0.6876 |

## Coverage Notes

- P1 remote columns separate standard 3-seed coverage (`42,123,456`, 15 jobs) from 5-seed coverage (`42,123,456,789,1024`, 25 jobs). Incomplete seed sets are left blank instead of averaged as complete evidence.
- P2 remote columns use the best complete 7-job run only: crop1-crop5, random, and worst must all be present.
- P3 remote columns use the best complete 5-outer run only: outer0/seed42 through outer4/seed46 must all be present.
- DEP/HC diagnostic CSVs are included as `model_deep_dive_csv`; subject-level hard-subject ledgers are excluded from the route score table because they are not route/protocol aggregate scores.
- Markdown diagnostic tables are included in the detail CSV when they expose route/run-level score columns; this keeps older scattered reports searchable without promoting them to canonical route status.
