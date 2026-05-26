# 2026-05-26 7top Promotion Evidence

本报告替换旧的 quota-router diagnostic 记录，记录今天真正要进入 promotion 讨论的 7 条 Top MoE 路线。报告只提交聚合表和方法说明；checkpoint、完整 score matrix、prediction table、raw outputs 仍在 Release/外部归档中，不进入 git。

## 结论

- 产出对象：`strict_top7_moe_01` 到 `strict_top7_moe_07`，共 7 条 promotion 路线。
- 路线来源：先用 2fold/seed42 的二阶段 MoE 初筛冻结 7 条候选定义；promotion 评估不再用初筛分数作最终结论。
- P1FULL：26 个唯一 component variant 全部重训，5 seeds x 5 folds，650 个 component-fold 矩阵齐全；router 25 个 P1 split 齐全；最终按 100 个 seed-fold-subject groups 汇总。
- P2：26 个 component 在 7 个 crop policy 上齐全，共 182 个 eval matrix；按 84 个 holdout policy-subject groups 汇总。
- P3：做 nested fusion/router validation，5 个 outer folds；inner 选择融合/router 权重，outer 评估 selected 权重。该 P3 阶段不重新搜索 backbone 架构。
- 最终排序：`score_04p1_04p2_02p3 = 0.4 * P1FULL + 0.4 * P2 + 0.2 * P3_selected`。
- 不改 route registry/status；这里是 promotion evidence bundle，不是直接修改仓库 route 状态。

## 最终 7top 排名

| final_rank | route_id | original_rank | P1FULL | P2 | P3_selected | P3_delta | final_score |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `strict_top7_moe_07` | 7 | 0.780250 | 0.758929 | 0.782712 | 0.002461 | 0.772214 |
| 2 | `strict_top7_moe_06` | 6 | 0.780250 | 0.758929 | 0.780636 | 0.000386 | 0.771799 |
| 3 | `strict_top7_moe_01` | 1 | 0.779066 | 0.752976 | 0.779736 | 0.000670 | 0.768764 |
| 4 | `strict_top7_moe_03` | 3 | 0.779161 | 0.747024 | 0.781414 | 0.002253 | 0.766756 |
| 5 | `strict_top7_moe_02` | 2 | 0.778484 | 0.741071 | 0.776879 | -0.001604 | 0.763198 |
| 6 | `strict_top7_moe_05` | 5 | 0.774186 | 0.741071 | 0.780873 | 0.006687 | 0.762278 |
| 7 | `strict_top7_moe_04` | 4 | 0.776285 | 0.720238 | 0.781245 | 0.004960 | 0.754858 |

## P1FULL 分数

P1FULL 口径：5 seeds `[42, 123, 456, 789, 1024]` x 5 folds；每条 route 的 DEP/HC expert 先用对应 component 重训分数矩阵融合，再用 P1 OOF router 做 hard/soft routing。

| route_id | P1FULL mean | DEP mean | HC mean | DEP-HC gap | min_group | groups |
|---|---:|---:|---:|---:|---:|---:|
| `strict_top7_moe_06` | 0.780250 | 0.696356 | 0.808215 | 0.111858 | 0.469328 | 100 |
| `strict_top7_moe_07` | 0.780250 | 0.696356 | 0.808215 | 0.111858 | 0.469328 | 100 |
| `strict_top7_moe_03` | 0.779161 | 0.698640 | 0.806001 | 0.107361 | 0.462128 | 100 |
| `strict_top7_moe_01` | 0.779066 | 0.697074 | 0.806396 | 0.109323 | 0.462128 | 100 |
| `strict_top7_moe_02` | 0.778484 | 0.697481 | 0.805484 | 0.108003 | 0.462128 | 100 |
| `strict_top7_moe_04` | 0.776285 | 0.686826 | 0.806104 | 0.119278 | 0.462128 | 100 |
| `strict_top7_moe_05` | 0.774186 | 0.693270 | 0.801158 | 0.107888 | 0.461920 | 100 |

## P2 分数

P2 口径：fresh holdout component/router artifacts；crop policies 为 `crop1..crop5/random/worst`，7 policy 全齐后再组装 7top。`min_group=0` 表示 worst/个别 policy-subject group 有崩点，不能只看 mean。

| route_id | P2 mean | DEP mean | HC mean | DEP-HC gap | min_group | groups |
|---|---:|---:|---:|---:|---:|---:|
| `strict_top7_moe_06` | 0.758929 | 0.723214 | 0.776786 | 0.053571 | 0.000000 | 84 |
| `strict_top7_moe_07` | 0.758929 | 0.723214 | 0.776786 | 0.053571 | 0.000000 | 84 |
| `strict_top7_moe_01` | 0.752976 | 0.732143 | 0.763393 | 0.031250 | 0.000000 | 84 |
| `strict_top7_moe_03` | 0.747024 | 0.723214 | 0.758929 | 0.035714 | 0.000000 | 84 |
| `strict_top7_moe_05` | 0.741071 | 0.687500 | 0.767857 | 0.080357 | 0.250000 | 84 |
| `strict_top7_moe_02` | 0.741071 | 0.714286 | 0.754464 | 0.040179 | 0.000000 | 84 |
| `strict_top7_moe_04` | 0.720238 | 0.642857 | 0.758929 | 0.116071 | 0.000000 | 84 |

## P3 分数

P3 口径：nested fusion/router validation。每个 outer fold 用其它 folds 做 inner selection；候选包括 base/uniform/one-hot/Dirichlet 权重、hard/soft router 和 raw/sigmoid transform。这里验证的是 fusion/router 权重调参收益，不把 outer test 参与 inner selection。

| route_id | baseline | selected | delta | selected DEP | selected HC | selected gap | min_outer | outer_units |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `strict_top7_moe_07` | 0.780250 | 0.782712 | 0.002461 | 0.698489 | 0.810786 | 0.112297 | 0.750920 | 5 |
| `strict_top7_moe_03` | 0.779161 | 0.781414 | 0.002253 | 0.699955 | 0.808567 | 0.108612 | 0.745702 | 5 |
| `strict_top7_moe_04` | 0.776285 | 0.781245 | 0.004960 | 0.694010 | 0.810323 | 0.116313 | 0.749172 | 5 |
| `strict_top7_moe_05` | 0.774186 | 0.780873 | 0.006687 | 0.698065 | 0.808476 | 0.110410 | 0.747225 | 5 |
| `strict_top7_moe_06` | 0.780250 | 0.780636 | 0.000386 | 0.695993 | 0.808851 | 0.112858 | 0.747197 | 5 |
| `strict_top7_moe_01` | 0.779066 | 0.779736 | 0.000670 | 0.699520 | 0.806474 | 0.106954 | 0.750456 | 5 |
| `strict_top7_moe_02` | 0.778484 | 0.776879 | -0.001604 | 0.694955 | 0.804187 | 0.109233 | 0.745280 | 5 |

## 路线定义与方法

每条 `strict_top7_moe_*` 都是两专家 MoE：DEP expert 和 HC expert 分开由 component score 融合得到；router 决定每个 subject/trial 更信 DEP 还是 HC。初筛阶段的 2fold 分数只用于冻结候选结构和初始权重，P1FULL/P2/P3 重新按严格口径组装。

### strict_top7_moe_01

- 初筛 rank：1；初筛 mean BA：0.837272；DEP：0.858576；HC：0.830171；min_group：0.720368。
- promotion 汇总：P1FULL `0.779066`，P2 `0.752976`，P3 selected `0.779736`，final `0.768764`。
- router：`p1_oof_quota_hard`；mode：`quota_hard_known_dep_count`；router transform：`raw`。

| expert | weight | component | variant | member_transform |
|---|---:|---|---:|---|
| DEP | 0.132000 | `fixed_crop_whitening_eps3e4_fbstcnet_m_conn_light_dep_mix75_ce` | 0 | `sigmoid_t1p25` |
| DEP | 0.296000 | `fixed_crop_ea_deformer_ff2_dep_w3_smooth005` | 0 | `sigmoid_t1p25` |
| DEP | 0.347000 | `fixed_crop_car_fbstcnet_train_channel_zscore_v1_dep_mix75_ce` | 1 | `sigmoid_t1p25` |
| DEP | 0.224000 | `fixed_crop_ea_dgcnn_region_scale_down_weak_v1_dep_w2_ce` | 0 | `sigmoid_t1p25` |
| HC | 0.146000 | `fixed_crop_car_fbstcnet_dc_shift_weak_v1_hc_w2_focal20` | 0 | `sigmoid_t1p0` |
| HC | 0.024000 | `fixed_crop_car_fbstcnet_channel_noise_v1_hc_w2_focal20` | 0 | `sigmoid_t1p0` |
| HC | 0.489000 | `fixed_crop_car_fbstcnet_freq_mask_weak_v1_hc_w3_ce` | 0 | `sigmoid_t1p0` |
| HC | 0.237000 | `sliding_ea_dgcnn_w6_s1_hc_w3_ce` | 1 | `sigmoid_t1p0` |
| HC | 0.103000 | `fixed_crop_car_fbstcnet_aug_light_v1_hc_w3_ce` | 1 | `sigmoid_t1p0` |

### strict_top7_moe_02

- 初筛 rank：2；初筛 mean BA：0.836216；DEP：0.854352；HC：0.830171；min_group：0.720368。
- promotion 汇总：P1FULL `0.778484`，P2 `0.741071`，P3 selected `0.776879`，final `0.763198`。
- router：`p1_oof_quota_hard`；mode：`quota_hard_known_dep_count`；router transform：`raw`。

| expert | weight | component | variant | member_transform |
|---|---:|---|---:|---|
| DEP | 0.466000 | `fixed_crop_ea_deformer_ff2_dep_w3_smooth005` | 0 | `raw` |
| DEP | 0.339000 | `fixed_crop_car_fbstcnet_train_channel_zscore_v1_dep_w3_smooth005` | 1 | `raw` |
| DEP | 0.155000 | `fixed_crop_whitening_eps3e4_fbstcnet_m_conn_light_dep_mix75_ce` | 0 | `raw` |
| DEP | 0.041000 | `fixed_crop_car_fbstcnet_crop_jitter_aug_light_v1_dep_resample3_ce` | 0 | `raw` |
| HC | 0.146000 | `fixed_crop_car_fbstcnet_dc_shift_weak_v1_hc_w2_focal20` | 0 | `sigmoid_t1p0` |
| HC | 0.024000 | `fixed_crop_car_fbstcnet_channel_noise_v1_hc_w2_focal20` | 0 | `sigmoid_t1p0` |
| HC | 0.489000 | `fixed_crop_car_fbstcnet_freq_mask_weak_v1_hc_w3_ce` | 0 | `sigmoid_t1p0` |
| HC | 0.237000 | `sliding_ea_dgcnn_w6_s1_hc_w3_ce` | 1 | `sigmoid_t1p0` |
| HC | 0.103000 | `fixed_crop_car_fbstcnet_aug_light_v1_hc_w3_ce` | 1 | `sigmoid_t1p0` |

### strict_top7_moe_03

- 初筛 rank：3；初筛 mean BA：0.836038；DEP：0.853640；HC：0.830171；min_group：0.720368。
- promotion 汇总：P1FULL `0.779161`，P2 `0.747024`，P3 selected `0.781414`，final `0.766756`。
- router：`p1_oof_quota_hard`；mode：`quota_hard_known_dep_count`；router transform：`raw`。

| expert | weight | component | variant | member_transform |
|---|---:|---|---:|---|
| DEP | 0.367000 | `fixed_crop_ea_deformer_ff2_dep_w2_ce` | 0 | `sigmoid_t0p5` |
| DEP | 0.411000 | `fixed_crop_car_fbstcnet_train_channel_zscore_v1_dep_mix75_ce` | 1 | `sigmoid_t0p5` |
| DEP | 0.059000 | `fixed_crop_dgcnn_noea_zscore_dep_w3_smooth005` | 0 | `sigmoid_t0p5` |
| DEP | 0.162000 | `fixed_crop_whitening_eps3e4_fbstcnet_m_conn_light_dep_mix75_ce` | 0 | `sigmoid_t0p5` |
| HC | 0.146000 | `fixed_crop_car_fbstcnet_dc_shift_weak_v1_hc_w2_focal20` | 0 | `sigmoid_t1p0` |
| HC | 0.024000 | `fixed_crop_car_fbstcnet_channel_noise_v1_hc_w2_focal20` | 0 | `sigmoid_t1p0` |
| HC | 0.489000 | `fixed_crop_car_fbstcnet_freq_mask_weak_v1_hc_w3_ce` | 0 | `sigmoid_t1p0` |
| HC | 0.237000 | `sliding_ea_dgcnn_w6_s1_hc_w3_ce` | 1 | `sigmoid_t1p0` |
| HC | 0.103000 | `fixed_crop_car_fbstcnet_aug_light_v1_hc_w3_ce` | 1 | `sigmoid_t1p0` |

### strict_top7_moe_04

- 初筛 rank：4；初筛 mean BA：0.836020；DEP：0.853568；HC：0.830171；min_group：0.720368。
- promotion 汇总：P1FULL `0.776285`，P2 `0.720238`，P3 selected `0.781245`，final `0.754858`。
- router：`p1_oof_quota_hard`；mode：`quota_hard_known_dep_count`；router transform：`raw`。

| expert | weight | component | variant | member_transform |
|---|---:|---|---:|---|
| DEP | 0.566000 | `fixed_crop_dgcnn_noea_zscore_dep_w3_focal15` | 0 | `sigmoid_t1p0` |
| DEP | 0.102000 | `fixed_crop_ea_deformer_ff2_dep_w3_smooth005` | 0 | `sigmoid_t1p0` |
| DEP | 0.055000 | `fixed_crop_whitening_eps3e4_fbstcnet_m_conn_light_dep_only_ce` | 0 | `sigmoid_t1p0` |
| DEP | 0.277000 | `fixed_crop_car_fbstcnet_robust_clip_train_mad8_v1_dep_resample3_ce` | 1 | `sigmoid_t1p0` |
| HC | 0.146000 | `fixed_crop_car_fbstcnet_dc_shift_weak_v1_hc_w2_focal20` | 0 | `sigmoid_t1p0` |
| HC | 0.024000 | `fixed_crop_car_fbstcnet_channel_noise_v1_hc_w2_focal20` | 0 | `sigmoid_t1p0` |
| HC | 0.489000 | `fixed_crop_car_fbstcnet_freq_mask_weak_v1_hc_w3_ce` | 0 | `sigmoid_t1p0` |
| HC | 0.237000 | `sliding_ea_dgcnn_w6_s1_hc_w3_ce` | 1 | `sigmoid_t1p0` |
| HC | 0.103000 | `fixed_crop_car_fbstcnet_aug_light_v1_hc_w3_ce` | 1 | `sigmoid_t1p0` |

### strict_top7_moe_05

- 初筛 rank：5；初筛 mean BA：0.833226；DEP：0.854632；HC：0.826091；min_group：0.730720。
- promotion 汇总：P1FULL `0.774186`，P2 `0.741071`，P3 selected `0.780873`，final `0.762278`。
- router：`p1_oof_rank_soft_t0p1`；mode：`rank_soft_known_dep_count`；router transform：`raw`。

| expert | weight | component | variant | member_transform |
|---|---:|---|---:|---|
| DEP | 0.193000 | `fixed_crop_car_fbstcnet_robust_clip_mad8_robust_zscore_v1_dep_mix75_ce` | 1 | `raw` |
| DEP | 0.095000 | `fixed_crop_whitening_eps1e3_fbstcnet_m_conn_light_dep_w3_ce` | 0 | `raw` |
| DEP | 0.440000 | `fixed_crop_dgcnn_noea_zscore_dep_w3_smooth005` | 0 | `raw` |
| DEP | 0.273000 | `fixed_crop_ea_dgcnn_region_scale_down_weak_v1_dep_w2_focal20` | 0 | `raw` |
| HC | 0.014000 | `fixed_crop_car_fbstcnet_crop_jitter_aug_light_v1_hc_w2_focal20` | 0 | `sigmoid_t0p75` |
| HC | 0.660000 | `fixed_crop_car_fbstcnet_freq_mask_weak_v1_hc_w3_focal15` | 0 | `sigmoid_t0p75` |
| HC | 0.181000 | `sliding_ea_dgcnn_w6_s1_hc_w3_ce` | 1 | `sigmoid_t0p75` |
| HC | 0.002000 | `fixed_crop_car_fbstcnet_time_shift_v1_hc_w3_focal15` | 0 | `sigmoid_t0p75` |
| HC | 0.143000 | `fixed_crop_car_fbstcnet_band_amp_scale_weak_v1_hc_w3_smooth005` | 0 | `sigmoid_t0p75` |

### strict_top7_moe_06

- 初筛 rank：6；初筛 mean BA：0.833266；DEP：0.858576；HC：0.824829；min_group：0.717328。
- promotion 汇总：P1FULL `0.780250`，P2 `0.758929`，P3 selected `0.780636`，final `0.771799`。
- router：`p1_oof_quota_hard`；mode：`quota_hard_known_dep_count`；router transform：`raw`。

| expert | weight | component | variant | member_transform |
|---|---:|---|---:|---|
| DEP | 0.132000 | `fixed_crop_whitening_eps3e4_fbstcnet_m_conn_light_dep_mix75_ce` | 0 | `sigmoid_t1p25` |
| DEP | 0.296000 | `fixed_crop_ea_deformer_ff2_dep_w3_smooth005` | 0 | `sigmoid_t1p25` |
| DEP | 0.347000 | `fixed_crop_car_fbstcnet_train_channel_zscore_v1_dep_mix75_ce` | 1 | `sigmoid_t1p25` |
| DEP | 0.224000 | `fixed_crop_ea_dgcnn_region_scale_down_weak_v1_dep_w2_ce` | 0 | `sigmoid_t1p25` |
| HC | 0.339000 | `fixed_crop_car_fbstcnet_dc_shift_weak_v1_hc_w3_ce` | 0 | `sigmoid_t1p25` |
| HC | 0.059000 | `fixed_crop_car_fbstcnet_crop_jitter_aug_light_v1_hc_w2_ce` | 0 | `sigmoid_t1p25` |
| HC | 0.355000 | `fixed_crop_car_fbstcnet_freq_mask_weak_v1_hc_w3_focal15` | 0 | `sigmoid_t1p25` |
| HC | 0.247000 | `sliding_ea_dgcnn_w6_s1_hc_w2_ce` | 1 | `sigmoid_t1p25` |

### strict_top7_moe_07

- 初筛 rank：7；初筛 mean BA：0.833266；DEP：0.858576；HC：0.824829；min_group：0.717328。
- promotion 汇总：P1FULL `0.780250`，P2 `0.758929`，P3 selected `0.782712`，final `0.772214`。
- router：`p1_oof_quota_hard`；mode：`quota_hard_known_dep_count`；router transform：`raw`。

| expert | weight | component | variant | member_transform |
|---|---:|---|---:|---|
| DEP | 0.132000 | `fixed_crop_whitening_eps3e4_fbstcnet_m_conn_light_dep_mix75_ce` | 0 | `sigmoid_t1p25` |
| DEP | 0.296000 | `fixed_crop_ea_deformer_ff2_dep_w3_smooth005` | 0 | `sigmoid_t1p25` |
| DEP | 0.347000 | `fixed_crop_car_fbstcnet_train_channel_zscore_v1_dep_mix75_ce` | 1 | `sigmoid_t1p25` |
| DEP | 0.224000 | `fixed_crop_ea_dgcnn_region_scale_down_weak_v1_dep_w2_ce` | 0 | `sigmoid_t1p25` |
| HC | 0.339000 | `fixed_crop_car_fbstcnet_dc_shift_weak_v1_hc_w3_ce` | 0 | `sigmoid_t1p25` |
| HC | 0.059000 | `fixed_crop_car_fbstcnet_crop_jitter_aug_light_v1_hc_w2_ce` | 0 | `sigmoid_t1p25` |
| HC | 0.355000 | `fixed_crop_car_fbstcnet_freq_mask_weak_v1_hc_w3_focal15` | 0 | `sigmoid_t1p25` |
| HC | 0.247000 | `sliding_ea_dgcnn_w6_s1_hc_w2_ce` | 1 | `sigmoid_t1p25` |

## 附带 CSV

| file | 内容 |
|---|---|
| `reports/model_deep_dive/top7_promotion_20260526/top7_protocol_scores.csv` | 最终 P1/P2/P3 合并排名 |
| `reports/model_deep_dive/top7_promotion_20260526/top7_moe_routes.csv` | 7 条 MoE route 的初筛定义、router、完整 label |
| `reports/model_deep_dive/top7_promotion_20260526/top7_component_membership.csv` | 每条 route 的 DEP/HC component 权重和 transform |
| `reports/model_deep_dive/top7_promotion_20260526/top7_unique_component_variants.csv` | 26 个唯一 component variant 的初筛指标摘要 |
| `reports/model_deep_dive/top7_promotion_20260526/top7_p1full_metrics.csv` | P1FULL 聚合分数 |
| `reports/model_deep_dive/top7_promotion_20260526/top7_p2_metrics.csv` | P2 聚合分数 |
| `reports/model_deep_dive/top7_promotion_20260526/top7_p2_policy_metrics.csv` | P2 每 crop policy 分数 |
| `reports/model_deep_dive/top7_promotion_20260526/top7_p3_nested_metrics.csv` | P3 nested 聚合分数 |
| `reports/model_deep_dive/top7_promotion_20260526/top7_p3_outer_selections.csv` | P3 outer fold selection 明细 |

## 证据边界

- Release 归档入口：`https://github.com/metaphordivine/hust-bci-er/releases/tag/v2026.05.26-remote-artifacts`。
- 本 PR 只提交可 review 的聚合表和文字说明，不提交 checkpoint、raw outputs、full score matrix、prediction table 或临时实验 dump。
- 没有使用 public/private labels、leaderboard feedback、split 修改，`subject_id/trial_id` 只用于 group 对齐和审计，不作为模型特征。
- 如果后续要把 7 条 route 写入 registry 并改成正式 promoted/candidate 状态，需要单独绑定 experiment audit gate；本 PR 当前只记录 promotion 证据链。
