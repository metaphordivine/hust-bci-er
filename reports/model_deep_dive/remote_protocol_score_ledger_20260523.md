# Remote Protocol Score Ledger 2026-05-23

Scope: A/B/C remote protocol summaries pulled from `scratch/remote_protocol_scores_20260523`. This ledger stores aggregate protocol-board rows only; it excludes checkpoints, raw predictions, and subject-level tables.

This is a score ledger snapshot, not route-status promotion evidence. Route promotion still requires the normal audit summary and `repo_doctor.py experiment` gate.

## Source Archives

| machine | archive | sha256 | generated_at | manifests_seen | complete_boards | errors |
|---|---|---|---|---:|---:|---:|
| A | `remote_protocol_scores_A.tgz` | `87d75a6bc9818fb7c6b9e2631c9ffb83016e353cae18d99be47b9fba7af70ffe` | 2026-05-23T23:51:59+0800 | 91 | 90 | 0 |
| B | `remote_protocol_scores_B.tgz` | `6f897340242462a081c466facc7b4ee8bbc58b4cf5cfede780627603324167ba` | 2026-05-23T23:51:22+0800 | 227 | 201 | 0 |
| C | `remote_protocol_scores_C.tgz` | `18dd01bbe5419107626747efd6ca05564ef48be5441007948df37ab433868b59` | 2026-05-23T23:51:21+0800 | 223 | 199 | 0 |

## Row Counts

| bucket | count |
|---|---:|
| machine A | 90 |
| machine B | 201 |
| machine C | 199 |
| `p1_repeated_group_kfold` | 164 |
| `p2_pseudo_public_holdout` | 123 |
| `p3_nested_selection` | 203 |

## Top Route Snapshot

| route_id | ledger_rows | best_mean | mean_of_means |
|---|---:|---:|---:|
| `fixed_crop_car_fbstcnet` | 8 | 0.7250 | 0.7134 |
| `fixed_crop_ea_whitening_eps3e4_fbstcnet_m_conn_light` | 29 | 0.7199 | 0.6893 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet_m_conn_light` | 11 | 0.7197 | 0.6924 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet_m_power_light` | 5 | 0.7190 | 0.6890 |
| `fixed_crop_whitening_eps3e4_fbstcnet_m_conn_light` | 42 | 0.7172 | 0.6951 |
| `fixed_crop_whitening_eps1e3_fbstcnet_m_conn_light` | 23 | 0.7168 | 0.6985 |
| `fixed_crop_ea_fbstcnet_c_only` | 17 | 0.7124 | 0.6979 |
| `fixed_crop_whitening_eps1e3_fbstcnet` | 25 | 0.7122 | 0.6913 |
| `fixed_crop_whitening_eps3e4_fbstcnet` | 20 | 0.7121 | 0.6980 |
| `fixed_crop_ea_fbstcnet_m_conn_light` | 16 | 0.7104 | 0.7011 |
| `fixed_crop_ea_whitening_eps3e4_fbstcnet` | 20 | 0.7103 | 0.6942 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet` | 18 | 0.7082 | 0.6931 |
| `fixed_crop_ea_whitening_eps3e4_fbstcnet_c_only` | 13 | 0.7075 | 0.6864 |
| `fixed_crop_ea_car_fbstcnet` | 1 | 0.7071 | 0.7071 |
| `fixed_crop_whitening_eps3e4_tri_context_gate` | 2 | 0.7056 | 0.6936 |
| `sliding_ea_dgcnn_coral_cohort_w6_s1` | 8 | 0.7054 | 0.6607 |
| `fixed_crop_ea_fbstcnet_riem_guided_gate` | 8 | 0.7043 | 0.6926 |
| `fixed_crop_ea_fbstcnet_m_power_light` | 3 | 0.7035 | 0.6941 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet_c_only` | 5 | 0.7031 | 0.6773 |
| `fixed_crop_ea_dgcnn` | 14 | 0.7016 | 0.6968 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet_p_only` | 6 | 0.6944 | 0.6714 |
| `fixed_crop_ea_fbstcnet_p_only` | 3 | 0.6938 | 0.6831 |
| `fixed_crop_ea_deformer_depth1_emb64` | 13 | 0.6878 | 0.6730 |
| `sliding_ea_dgcnn_w6_s1` | 12 | 0.6875 | 0.6465 |
| `fixed_crop_ea_deformer_depth2_emb96` | 5 | 0.6804 | 0.6804 |
| `fixed_crop_ea_deformer_depth3_emb64` | 7 | 0.6772 | 0.6772 |
| `sliding_ea_fbstcnet_m_conn_w8_s0p5` | 4 | 0.6726 | 0.6613 |
| `sliding_window_srf_fbstcnet_gate_whitening_eps3e4_w6_s1` | 12 | 0.6714 | 0.6563 |
| `sliding_ea_dgcnn_dann_cohort_w6_s1` | 7 | 0.6667 | 0.6547 |
| `sliding_ea_dgcnn_masked_consistency_w6_s1` | 3 | 0.6637 | 0.6457 |

Full CSV: `reports/model_deep_dive/remote_protocol_score_ledger_20260523.csv`

Clean route-level P1/P2/P3 table: `reports/model_deep_dive/remote_protocol_route_score_table_20260523.md`

All repository evaluation score table: `reports/model_deep_dive/all_evaluation_score_table_20260524.md`
