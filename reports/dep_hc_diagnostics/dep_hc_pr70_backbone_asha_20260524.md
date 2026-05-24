# PR70 DEP/HC Backbone ASHA 2026-05-24
本报告汇总机器 B 在 PR70 worktree 上运行的 DEP/HC classifier ASHA 诊断。所有数值来自 P3 口径的 diagnostic run，使用 `crop_combo_expected_*` 指标；不是 candidate evidence，不包含 raw prediction tables 或 checkpoints。
## 运行批次
| group | 内容 | 备注 |
|---|---|---|
| `deformer_reference` | deformer_lite 参数/校准 ASHA | 早前同一 PR70 会话的参考基线，合并 initial rung2 与 resume confirm。 |
| `backbone` | FBSTCNet/FBCNet/TSception/Shallow/EEGNet/SRF/Conformer/Graph mixed ASHA | 27 candidates, 6→12→22 epoch, top3 confirm。 |
| `hybrid_resume` | SRF+FBST gate、Tri-context gate、dual graph conformer、CBraMod、deformer/conformer controls | 修正 CBraMod patch250 维度后恢复；top2 confirm。 |
| `topbackbone` | TSception/FBCNet/FBST-FFT 的 calibration/threshold/sampling 聚焦 ASHA | 重点看 min-recall、Platt、subject-balanced 是否稳。 |
| `toppreproc` | Top TSception/FBC/FBST-FFT 的 raw / zscore-only 预处理消融 ASHA | 对齐“删预处理”问题。 |

## 主要结论
- 非 deformer backbone 中，`tsc_f8_k31_63_drop045` 是当前最稳的完整 confirm：20 rows mean expected BA `0.7610`，但 worst-min 只有 `0.3125`，仍有弱 split。
- 这批非-deformer 的最好完整 confirm 均值略高于本次 deformer ASHA reference（`conv24_emb48_lr7e4` 20 rows mean `0.7267`），但 worst-min 不占优，因此更适合作为融合互补分支。
- `topbackbone` 的 min-recall 调参没有超过原始 `tsc_f8_k31_63_drop045`：`tsc8_minrec_vote_b160` 20 rows mean `0.7358`。阈值目标能改善局部 recall，但没有整体更稳。
- hybrid/gate 初筛很亮眼，但 full confirm 回落：`tri_context_wide_vote` 20 rows mean `0.7213`，`srf_fbst_gate_wide_vote` 20 rows mean `0.6896`。它们更适合保留为融合/解释组件，而不是替代主 backbone。
- `raw` / `zscore-only` 预处理消融没有给出稳定增益：`raw_tsc4_minrec_vote_b192` 20 rows mean `0.6833`。删除 CAR 在短 epoch 有局部高点，但 P3 confirm 不稳。
- Graph/CBraMod 在这批 ASHA 中没有晋级；`dual_graph_conformer` 和 `CBraMod` 只适合低优先级诊断或融合弱分支。

## Full / Near-Full Confirm Summary
| group | candidate | model | n | expected BA mean | expected BA min | worst mean | worst min | min-recall mean | HC recall mean | DEP recall mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `backbone` | `tsc_f8_k31_63_drop045` | `tsception` | 20 | 0.7610 | 0.5335 | 0.5629 | 0.3125 | 0.6588 | 0.7952 | 0.8083 |
| `topbackbone` | `tsc8_minrec_vote_b160` | `tsception` | 20 | 0.7358 | 0.5163 | 0.5395 | 0.3364 | 0.6304 | 0.7917 | 0.7550 |
| `backbone` | `tsc_f4_k31_63_vote` | `tsception` | 20 | 0.7315 | 0.5108 | 0.5493 | 0.3500 | 0.5912 | 0.7467 | 0.7858 |
| `deformer_reference` | `p01_vote_platt_window` | `deformer_lite` | 20 | 0.7301 | 0.5769 | 0.5315 | 0.3500 | 0.5812 | 0.7402 | 0.7683 |
| `deformer_reference` | `conv24_emb48_lr7e4` | `deformer_lite` | 20 | 0.7267 | 0.5335 | 0.5354 | 0.3364 | 0.5859 | 0.7648 | 0.7342 |
| `hybrid_resume` | `tri_context_wide_vote` | `tri_context_gate` | 20 | 0.7213 | 0.5630 | 0.5424 | 0.3750 | 0.5799 | 0.7388 | 0.7458 |
| `deformer_reference` | `conv32_emb48_vote` | `deformer_lite` | 20 | 0.7210 | 0.5876 | 0.5500 | 0.4000 | 0.5991 | 0.7506 | 0.7292 |
| `topbackbone` | `tsc4_minrec_vote_b192` | `tsception` | 20 | 0.7193 | 0.5392 | 0.5324 | 0.3364 | 0.6014 | 0.7993 | 0.7233 |
| `hybrid_resume` | `srf_fbst_gate_wide_vote` | `srf_fbstcnet_gate` | 20 | 0.6896 | 0.5037 | 0.5186 | 0.3364 | 0.5329 | 0.7217 | 0.7208 |
| `toppreproc` | `raw_tsc4_minrec_vote_b192` | `tsception` | 20 | 0.6833 | 0.4953 | 0.5228 | 0.3125 | 0.5771 | 0.7247 | 0.6908 |
| `backbone` | `fbst_m_fft_6b_g200_vote` | `fbstcnet` | 20 | 0.6688 | 0.4825 | 0.5287 | 0.3000 | 0.5266 | 0.7239 | 0.6933 |

## Top Diagnostic Rows
| group | candidate | model | n | stages | expected BA mean | worst mean | min-recall mean | setting |
|---|---|---|---:|---|---:|---:|---:|---|
| `backbone` | `tsc_f8_k31_63_drop045` | `tsception` | 20 | `confirm_inner,confirm_outer,rung2` | 0.7610 | 0.5629 | 0.6588 | `temperature/balanced_accuracy/vote_frac` |
| `topbackbone` | `tsc4_platt_vote_b192` | `tsception` | 1 | `rung2` | 0.7403 | 0.4470 | 0.6729 | `platt/balanced_accuracy/vote_frac` |
| `topbackbone` | `tsc8_minrec_vote_b160` | `tsception` | 20 | `confirm_inner,confirm_outer,rung2` | 0.7358 | 0.5395 | 0.6304 | `temperature/min_recall/vote_frac` |
| `backbone` | `tsc_f4_k31_63_vote` | `tsception` | 20 | `confirm_inner,confirm_outer,rung2` | 0.7315 | 0.5493 | 0.5912 | `temperature/balanced_accuracy/vote_frac` |
| `deformer_reference` | `p01_vote_platt_window` | `deformer_lite` | 20 | `confirm_inner,confirm_outer,rung2` | 0.7301 | 0.5315 | 0.5812 | `platt/balanced_accuracy/vote_frac` |
| `deformer_reference` | `conv24_emb48_lr7e4` | `deformer_lite` | 20 | `confirm_inner,confirm_outer,rung2` | 0.7267 | 0.5354 | 0.5859 | `temperature/balanced_accuracy/vote_frac` |
| `hybrid_resume` | `tri_context_wide_vote` | `tri_context_gate` | 20 | `confirm_inner,confirm_outer,rung2` | 0.7213 | 0.5424 | 0.5799 | `temperature/balanced_accuracy/vote_frac` |
| `deformer_reference` | `conv32_emb48_vote` | `deformer_lite` | 20 | `confirm_inner,confirm_outer,rung2` | 0.7210 | 0.5500 | 0.5991 | `temperature/balanced_accuracy/vote_frac` |
| `topbackbone` | `tsc4_minrec_vote_b192` | `tsception` | 20 | `confirm_inner,confirm_outer,rung2` | 0.7193 | 0.5324 | 0.6014 | `temperature/min_recall/vote_frac` |
| `toppreproc` | `zscore_tsc8_platt_vote_b160` | `tsception` | 1 | `rung2` | 0.6947 | 0.5303 | 0.5880 | `platt/balanced_accuracy/vote_frac` |
| `hybrid_resume` | `srf_fbst_gate_wide_vote` | `srf_fbstcnet_gate` | 20 | `confirm_inner,confirm_outer,rung2` | 0.6896 | 0.5186 | 0.5329 | `temperature/balanced_accuracy/vote_frac` |
| `toppreproc` | `raw_tsc4_minrec_vote_b192` | `tsception` | 20 | `confirm_inner,confirm_outer,rung2` | 0.6833 | 0.5228 | 0.5771 | `temperature/min_recall/vote_frac` |

## Resource Notes
| run | train samples | avg GPU util | avg mem MB | max mem MB |
|---|---:|---:|---:|---:|
| `backbone` | 38 | 97.6 | 9098.3 | 18100 |
| `hybrid_resume` | 78 | 97.1 | 8323.0 | 18483 |
| `topbackbone` | 38 | 99.9 | 12668.6 | 18483 |
| `toppreproc` | 12 | 79.5 | 6439.4 | 10121 |

## Next Actions
- 把 `tsc_f8_k31_63_drop045` 作为非-deformer 对照候选，优先做与当前 deformer default 的 score-level fusion，而不是继续扩大 TSception 单模型。
- hybrid/gate 只保留 top 配置进入 fusion 或 hard-subject disagreement 诊断；不要按初筛高点直接提升为主路线。
- 预处理消融目前支持保留 `car+zscore` 默认；raw/zscore-only 可作为弱分支但不应替换默认。

Compact TSV: `reports/dep_hc_diagnostics/dep_hc_pr70_backbone_asha_20260524_summary.tsv`
