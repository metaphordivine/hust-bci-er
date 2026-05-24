# Remote Protocol Route Score Table 2026-05-24

Scope: combined route-level aggregation of all committed remote protocol score ledgers through the E-machine >0.72 backfill.

Full CSV: `reports/model_deep_dive/remote_protocol_route_score_table_20260524.csv`

## P1 5-Seed Leaders

| route | P1 seeds | P1 mean | P2 best | P3 best |
|---|---|---|---|---|
| `fixed_crop_car_fbstcnet` | 42,123,456,789,1024 | 0.7146 | 0.6756 | 0.7250 |
| `fixed_crop_whitening_eps3e4_fbstcnet_m_conn_light` | 42,123,456,789,1024 | 0.7102 | 0.6548 | 0.7128 |
| `fixed_crop_whitening_eps1e3_fbstcnet_m_conn_light` | 42,123,456,789,1024 | 0.7099 | 0.6518 | 0.7127 |
| `fixed_crop_whitening_eps3e4_fbstcnet` | 42,123,456,789,1024 | 0.7097 | 0.6607 | 0.7121 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet_m_conn_light` | 42,123,456,789,1024 | 0.7086 | 0.6577 | 0.7149 |
| `fixed_crop_whitening_eps1e3_fbstcnet` | 42,123,456,789,1024 | 0.7085 | 0.6577 | 0.7122 |
| `fixed_crop_ea_whitening_eps3e4_fbstcnet_m_conn_light` | 42,123,456,789,1024 | 0.7081 | 0.6518 | 0.7116 |
| `fixed_crop_ea_whitening_eps3e4_fbstcnet` | 42,123,456,789,1024 | 0.7080 | 0.6518 | 0.7085 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet` | 42,123,456,789,1024 | 0.7075 | 0.6548 | 0.7082 |
| `fixed_crop_whitening_eps3e4_tri_context_gate` | 42,123,456,789,1024 | 0.7056 | 0.6815 |  |
| `fixed_crop_ea_fbstcnet_c_only` | 42,123,456,789,1024 | 0.7014 | 0.6696 | 0.7124 |
| `fixed_crop_ea_whitening_eps3e4_fbstcnet_c_only` | 42,123,456,789,1024 | 0.7005 | 0.6429 | 0.7018 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet_c_only` | 42,123,456,789,1024 | 0.7000 | 0.6488 | 0.7031 |
| `fixed_crop_ea_fbstcnet_m_conn_light` | 42,123,456,789,1024 | 0.6992 | 0.6845 | 0.7104 |
| `fixed_crop_ea_fbstcnet_m_power_light` | 42,123,456,789,1024 | 0.6942 | 0.6845 | 0.7035 |

## P2 7-Job Leaders

| route | P2 runs | P2 best | P1 5seed | P3 best |
|---|---|---|---|---|
| `sliding_ea_dgcnn_coral_cohort_w6_s1` | 1 | 0.7054 | 0.6319 | 0.6345 |
| `sliding_ea_dgcnn_w6_s1` | 1 | 0.6875 | 0.6316 | 0.6282 |
| `fixed_crop_ea_dgcnn` | 1 | 0.6845 | 0.6932 | 0.7016 |
| `fixed_crop_ea_fbstcnet_m_conn_light` | 1 | 0.6845 | 0.6992 | 0.7104 |
| `fixed_crop_ea_fbstcnet_m_power_light` | 1 | 0.6845 | 0.6942 | 0.7035 |
| `fixed_crop_whitening_eps3e4_tri_context_gate` | 1 | 0.6815 | 0.7056 |  |
| `fixed_crop_ea_fbstcnet_p_only` | 1 | 0.6786 | 0.6770 | 0.6938 |
| `fixed_crop_ea_fbstcnet_riem_guided_gate` | 1 | 0.6786 | 0.6883 | 0.7043 |
| `fixed_crop_car_fbstcnet` | 1 | 0.6756 | 0.7146 | 0.7250 |
| `sliding_ea_fbstcnet_m_conn_w8_s0p5` | 1 | 0.6726 | 0.6448 |  |
| `fixed_crop_ea_fbstcnet_c_only` | 1 | 0.6696 | 0.7014 | 0.7124 |
| `sliding_ea_dgcnn_dann_cohort_w6_s1` | 1 | 0.6667 | 0.6407 | 0.6474 |
| `sliding_ea_dgcnn_masked_consistency_w6_s1` | 1 | 0.6637 | 0.6408 | 0.6327 |
| `fixed_crop_whitening_eps3e4_fbstcnet` | 1 | 0.6607 | 0.7097 | 0.7121 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet_m_conn_light` | 1 | 0.6577 | 0.7086 | 0.7149 |

## P3 5-Outer Leaders

| route | P3 runs | P3 best | P1 5seed | P2 best |
|---|---|---|---|---|
| `fixed_crop_car_fbstcnet` | 1 | 0.7250 | 0.7146 | 0.6756 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet_m_conn_light` | 1 | 0.7149 | 0.7086 | 0.6577 |
| `fixed_crop_whitening_eps3e4_fbstcnet_m_conn_light` | 1 | 0.7128 | 0.7102 | 0.6548 |
| `fixed_crop_whitening_eps1e3_fbstcnet_m_conn_light` | 1 | 0.7127 | 0.7099 | 0.6518 |
| `fixed_crop_ea_fbstcnet_c_only` | 1 | 0.7124 | 0.7014 | 0.6696 |
| `fixed_crop_whitening_eps1e3_fbstcnet` | 1 | 0.7122 | 0.7085 | 0.6577 |
| `fixed_crop_whitening_eps3e4_fbstcnet` | 1 | 0.7121 | 0.7097 | 0.6607 |
| `fixed_crop_ea_whitening_eps3e4_fbstcnet_m_conn_light` | 1 | 0.7116 | 0.7081 | 0.6518 |
| `fixed_crop_ea_fbstcnet_m_conn_light` | 1 | 0.7104 | 0.6992 | 0.6845 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet_m_power_light` | 1 | 0.7103 |  | 0.6548 |
| `fixed_crop_ea_whitening_eps3e4_fbstcnet` | 1 | 0.7085 | 0.7080 | 0.6518 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet` | 1 | 0.7082 | 0.7075 | 0.6548 |
| `fixed_crop_ea_car_fbstcnet` | 1 | 0.7071 |  |  |
| `fixed_crop_ea_fbstcnet_riem_guided_gate` | 1 | 0.7043 | 0.6883 | 0.6786 |
| `fixed_crop_ea_fbstcnet_m_power_light` | 1 | 0.7035 | 0.6942 | 0.6845 |
