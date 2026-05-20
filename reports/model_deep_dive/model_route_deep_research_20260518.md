# HUST BCI ER 模型路线深度调研报告

生成日期：2026-05-18

范围：本仓库 `route_registry`、`route_board`、模型工厂、route summaries、candidate diagnostics，以及公开 EEG emotion recognition / BCI 文献。

约束：不使用 public/private label 或 leaderboard feedback；不改变 split/evaluation protocol；本文只做路线分析和后续建议，不变更 route status。

## 0. 结论先行

本仓库当前真正的问题不是“缺一个更大的 backbone”，而是模型路线已经进入 evidence management 阶段：route 很多、单 split 高分很多，但 formal candidate 证据、cohort 诊断、crop 压力测试、score-fusion 语义、foundation provenance 还没有完全统一。

最值得优先推进的不是再开大量近邻路线，而是：

1. 先把 `fixed_crop_ea_dgcnn` 重新审计，把 `fixed_crop_ea_fbcnet*` 补齐，清理 `TBD` summary。
2. 以 `fixed_crop_ea_fbstcnet` 为 anchor，把 `fixed_crop_car_fbstcnet`、`fixed_crop_ea_whitening_eps1e3_fbstcnet`、`fixed_crop_ea_fbstcnet_c_only` 做 P1/P2 级确认。
3. 把 DEP/HC gap 作为硬诊断；当前最高分路线普遍 HC 强、DEP 弱，单看 overall BA 会误导。
4. Graph/domain-adaptation 应成为第二主线，foundation model 只能在有真实预训练权重、channel mapping、normalization provenance 后再进入候选路线。

## 1. 仓库当前模型路线盘点

截至本次读取，`configs/routes/models/` 中共有 62 个 route config。

| 维度 | 统计 |
|---|---:|
| 总 route config | 62 |
| `CANDIDATE` | 17 |
| `IDEA` | 45 |
| `single_split_only_pending_multisplit` | 29 |
| `missing_real_candidate_run` | 14 |
| 无 blocker | 17 |

按模型族粗分：

| model family | route 数 |
|---|---:|
| `fbstcnet` | 15 |
| `score_fusion` | 13 |
| `deformer_lite` | 13 |
| `cbramod` | 7 |
| `shallow_conv_net` | 4 |
| `fbcnet` | 2 |
| `tsception` | 2 |
| `dgcnn` | 1 |
| `lggnet` | 1 |
| `conformer_lite` | 1 |
| `eegnet` | 1 |
| `srfnet` | 1 |
| `toy_centroid` | 1 |

按预处理/增强粗分，路线主要集中在 `euclidean_alignment,zscore`、`zscore`、`whitening_eps1e3` 和 `split_first_fixed_crops` / `split_first_sliding_window`。这说明当前仓库并不是单纯在试 backbone，而是在同时探索 reference scheme、alignment、whitening、fixed crop、sliding window、connectivity、score fusion。

### 1.1 当前 formal candidate 层

| route | metric | status | 备注 |
|---|---:|---|---|
| `fixed_crop_ea_fbstcnet` | 0.686144 | CANDIDATE | 当前 formal anchor |
| `sliding_window_srfnet` | 0.659627 | CANDIDATE | sliding-window 强基线 |
| `whitening_eps1e3_with_conformer_srfnet_reference_average` | 0.657988 | CANDIDATE | fusion/reference average |
| `whitening_eps1e3_query_eps3e4_srfnet_context_fusion` | 0.649681 | CANDIDATE | context fusion |
| `conformer_srfnet_whitening_eps1e3_average` | 0.643967 | CANDIDATE | conformer/srfnet fusion |
| `sliding_window_eegnet` | 0.637912 | CANDIDATE | compact CNN baseline |
| `fixed_crop_ea_deformer` | 0.629530 | CANDIDATE | scratch/lite transformer 路线 |

### 1.2 当前高分但未完全 formal 的 IDEA 层

| route | metric | blocker / audit |
|---|---:|---|
| `fixed_crop_car_fbstcnet` | 0.706457 | `single_split_only_pending_multisplit` |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet` | 0.703641 | `single_split_only_pending_multisplit` |
| `fixed_crop_ea_fbstcnet_c_only` | 0.702109 | `single_split_only_pending_multisplit` |
| `fbstcnet_srfnet_whitening_eps1e3_average` | 0.699401 | `single_split_only_pending_multisplit` |
| `fixed_crop_fbstcnet_zscore_only` | 0.695777 | `single_split_only_pending_multisplit` |
| `fixed_crop_whitening_eps1e3_fbstcnet` | 0.694549 | `single_split_only_pending_multisplit` |
| `fixed_crop_ea_dgcnn` | 0.693190 | `PENDING_REAUDIT` |

这里最关键的事实是：最高分不在 formal candidate 层，而在 single-split / pending audit 层。后续路线选择必须先把证据层级拉齐，否则会被单 split 噪声、crop lucky case 或未完成审计误导。

## 2. 本仓库模型路线存在的问题

### 2.1 证据层级混杂

`fixed_crop_ea_fbstcnet` 是当前最稳的 formal anchor，但多个 IDEA 路线单 split 分数已经超过它。这个状态本身不坏，说明探索有效；问题是 route board 很容易把 formal evidence 和 exploratory evidence 放在同一视野里，导致“看起来最高分”与“可以推进 candidate”不是一回事。

建议 route board 之后按三层展示：

1. **formal candidate**：完成 candidate audit，可用于主线对比。
2. **promising single-split**：有高分，但必须补 P1/P2/P3 或 re-audit。
3. **placeholder / pending remote**：有 config 或 summary 占位，但无可比较 metric。

### 2.2 route 数量已经超过假设管理能力

62 个 model route 中，FBSTCNet、score fusion、deformer_lite 占据大多数。许多 route 是相近假设的局部变体：EA vs CAR vs whitening、P-only vs C-only、fixed crop vs sliding window、reference average vs context fusion。继续平铺新增 route 会让路线板变成实验日志，而不是决策工具。

更合适的组织方式是按 hypothesis family 收敛。例如：

```text
family: fbstcnet_fixed_crop_preproc_branch
anchor: fixed_crop_ea_fbstcnet
current_best_single_split: fixed_crop_car_fbstcnet
formal_next: P1 + P2 crop stress
stop_condition: P1 mean < fixed_crop_ea_fbstcnet by >0.01 or DEP gap worsens
```

这样可以避免每次看到新 metric 就继续开一条近邻 route。

### 2.3 最高分来自预处理和连接性，不是“更大模型”

FBSTCNet 批次中，`fixed_crop_ea_fbstcnet_c_only` 达到 `0.702109`，高于 mixed baseline `0.686144`；`fixed_crop_car_fbstcnet` 达到 `0.706457`；`fixed_crop_ea_whitening_eps1e3_fbstcnet` 达到 `0.703641`。这说明当前数据上最有价值的方向至少包括：

1. reference/average 方式：CAR、whitening、EA 的交互。
2. connectivity branch：C-only 强于 P-only，说明通道间关系比单纯 band power 更关键。
3. fixed crop policy：FBSTCNet 在 5x10s fixed crops 下明显强于 sliding-window FBSTCNet。

外部文献也支持这个判断：FBSTCNet 本身就是把 power 与 connectivity 两类特征纳入 filter-bank spatio-temporal convolution 的 emotion decoding 模型，DBLP 记录其发表于 IEEE Transactions on Affective Computing 2024，题名即强调 power/connectivity integration。LGGNet、TSception、近年的 graph/domain-adaptation 工作也都强调时空、脑区、连接结构，而不是单纯加深 backbone。

### 2.4 DEP/HC cohort gap 是目前最严重的科学问题

高分路线的 deep dive 中，DEP 与 HC 差距非常大：

| route | overall | DEP exact BA | HC exact BA | 差距 |
|---|---:|---:|---:|---:|
| `fixed_crop_car_fbstcnet` | 0.706457 | 0.560967 | 0.779202 | 0.218235 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet` | 0.703641 | 0.519012 | 0.795956 | 0.276944 |
| `fixed_crop_ea_fbstcnet_c_only` | 0.702109 | 0.551327 | 0.777500 | 0.226173 |
| `fbstcnet_srfnet_whitening_eps1e3_average` | 0.699401 | 0.473244 | 0.812480 | 0.339236 |

这不是普通随机波动。它意味着当前高分可能主要由 HC 侧提升驱动，DEP 侧仍然不稳。后续路线如果只优化 overall BA，会继续把问题藏起来。

必须新增固定诊断：

1. 每个 candidate/P1 summary 必须输出 DEP/HC mean、min、std。
2. P2 holdout 必须报告 cohort-wise crop sensitivity。
3. 不允许用 `subject_id`、文件名或 cohort label 作为 feature，但可以把 cohort 作为审计分组。
4. domain adaptation 或 source selection 只能在训练侧估计，不得从 test subjects 泄漏统计量。

### 2.5 crop policy 仍然有压力

FBSTCNet 批次 crop 表显示，同一路线不同 crop 的 BA 可相差 0.06 到 0.10。例如：

| route | best crop | worst crop | 观察 |
|---|---:|---:|---|
| `fixed_crop_car_fbstcnet` | 0.750000 | 0.666667 | crop_1 最好，crop_0 最差 |
| `fixed_crop_ea_whitening_eps1e3_fbstcnet` | 0.750000 | 0.645833 | 早期 crop 弱 |
| `fixed_crop_ea_fbstcnet_c_only` | 0.708333 | 0.645833 | crop_3/4 强，crop_1 弱 |

仓库主指标 `exact_single_crop_expected_BA` 是必要的，但 P2 中的 `worst`、`random`、`crop1-5` 压力测试还没有成为高分路线的默认门槛。外部 EEG 评估经验也说明，segment/window 级别评估很容易高估真实泛化，必须用 subject/group 级别 split 和严格 crop policy 约束。

### 2.6 score fusion 的分数语义需要隔离

`fbstcnet_srfnet_whitening_eps1e3_average` 的 deep dive 显示：

```text
mean = 0.000000
std = 0.790858
min = -1.298763
max = 1.720953
outside_0_1 = 0.645833
```

这说明 fusion score 是 ranking/logit-like score，不是校准概率。用它算 Top-4 或 exact BA 可以，但不能混同为概率，也不能和单模型 `y_score in [0,1]` 直接做 calibration/brier 对比。

建议：

1. score fusion summary 固定标注 `score_semantics: ranking_score` 或 `probability_score`。
2. fusion 前对 component score 做 train/val only 标准化或温度校准。
3. 用新 champion component 重建 fusion，例如 CAR FBSTCNet、C-only FBSTCNet、DGCNN P1 component，而不是继续围绕旧 `fixed_crop_ea_fbstcnet`。
4. fusion 权重进入 P3 nested selection，避免在同一 test split 上看分数调权重。

### 2.7 foundation model 路线没有真正使用 foundation evidence

本仓库 `src/hust_bci_er/models/backbones/cbramod.py` 实现了 criss-cross transformer 和 patch embedding，但没有看到预训练 checkpoint 加载、tokenizer provenance 或外部预训练权重审计。换言之，当前 CBraMod route 更像“CBraMod-style scratch model”，不是文献意义上的 EEG foundation model。

外部趋势很清楚：

1. BENDR 用 self-supervised objective 学 raw EEG 表征，强调跨硬件、跨 subject、跨 task fine-tune。
2. LaBraM 通过 channel patches、neural tokenizer、masked code prediction，在约 2,500 小时 EEG 上预训练。
3. CBraMod 通过 patch-based masked reconstruction 和 criss-cross attention，强调空间/时间依赖分开建模，并在 10 个 downstream BCI tasks / 12 个数据集上评估。
4. REVE 进一步强调任意长度和 electrode arrangement 的适配，在 92 个数据集、25,000 subjects、60,000+ 小时 EEG 上预训练。

因此，如果仓库要做 foundation 路线，最重要的不是再开一个 scratch Transformer，而是建立：

```text
pretrained_weight_source
pretraining_dataset_scope
input_channel_mapping
sample_rate/window_adapter
frozen/fine-tuned layer policy
no-test-statistics normalization proof
candidate manifest provenance
```

没有这些字段，foundation route 的名字会误导评估。

## 3. 外部调研要点

### 3.1 数据集和任务形态

TorchEEG 的 `BCI2022Dataset` 描述了与本仓库高度相近的任务形态：2022 EMOTION_BCI 竞赛关注 cross-subject emotion recognition，包含 80 名 participants，30 通道 EEG，250 Hz，28 个 video clips，标签包括 valence 和离散情绪。其示例同时支持 DE grid、raw 30x250 chunk、GNN graph 三类输入。这说明本仓库同时保留 raw CNN、filterbank、graph、score fusion 是合理的。

DEAP 和 SEED 系列是最常见外部参照：

| 数据集 | 关键点 | 对本仓库的启发 |
|---|---|---|
| BCI2022 / EMOTION_BCI | 30 EEG channels, 250 Hz, cross-individual emotion recognition | 本仓库应优先 subject-level generalization，不应只看 trial/crop 内部表现 |
| DEAP | 32 participants, 40 one-minute music videos, EEG + peripheral signals, valence/arousal ratings | trial-level emotion label 有时间噪声，crop/worst-crop 诊断重要 |
| SEED/SEED-IV/SEED-V | 电影片段诱发情绪，逐步扩展类别，常用于跨 subject/session | domain adaptation、graph、DE/connectivity 特征是主流方向 |

### 3.2 评估风险：subject leakage 与 dataset shift

Frontiers 2024 的 translational EEG data leakage 研究指出，segment-based holdout 会让同一 subject 的片段同时出现在 train/test 中，从而显著高估 unseen-subject 表现。该文直接比较 segment-based 与 subject-based holdout，结论是 segment-based 会强烈高估新 subject 性能。

Neurocomputing 2024 的 EEG emotion generalization 综述强调，EEG non-stationarity 会导致 dataset shift，cross-subject/cross-session 泛化需要 transfer learning、adaptive feature extraction、domain adaptation 等方法。

本仓库已经有 split-first、subject group split、P1/P2/P3 协议，这是正确方向。现在的问题不是没有防线，而是许多高分结果还停在单 split 或待审计层级，没有完成 formal protocol 证据。

### 3.3 模型架构趋势

| 路线 | 文献趋势 | 本仓库状态 | 判断 |
|---|---|---|---|
| EEGNet/ShallowConvNet | compact CNN，适合作强 baseline | sliding candidate 中等或偏弱 | 保留 baseline，不再投入大量调参 |
| FBCNet/FBSTCNet | filterbank + spectro-spatial/connectivity aggregation | FBSTCNet 当前最强，FBCNet P1 部分完成 | 主线应继续围绕 connectivity/filterbank 收敛 |
| TSception | 多尺度 temporal + asymmetric spatial | route 已有，P1 metric TBD | 值得完成 P1，而不是只占位 |
| LGGNet/DGCNN | local/global graph、动态 adjacency、脑区先验 | DGCNN P1 已有强 aggregate，LGGNet TBD | 图路线值得重审和补齐 |
| EEG Conformer/Deformer | CNN + Transformer，兼顾局部和长程依赖 | lite/scratch 变体结果不如 FBSTCNet | 不应盲目加深，需结合预训练或 graph/DA |
| Foundation models | BENDR/LaBraM/CBraMod/REVE 强调大规模预训练和跨设置适配 | 当前 CBraMod 是 scratch-style | 若无预训练 provenance，不应当作 foundation 优势 |
| Domain adaptation/self-supervised | DANN-MAT、MS-DCDA、EGDA、SCMM 等都围绕跨 subject/session/corpus | 仓库有 `adaptation/` 基础，但 route 很少真正使用 | 下一阶段高价值方向 |

### 3.4 最新趋势对本仓库的具体启发

2024 到 2026 的 EEG emotion recognition 文献趋势基本集中在五类：

1. **Graph + topology**：DGCNN、LGGNet、graph-guided DA、adaptive graph attention，用电极拓扑或功能连接建模。
2. **Domain adaptation**：DANN、多源动态 domain adaptation、graph-guided session alignment，针对 subject/session shift。
3. **Self-supervised / contrastive / masked modeling**：BENDR、SCMM、cascaded SSL，用未标注或跨 corpus 信号缓解小样本。
4. **Hybrid CNN/RNN/Transformer**：不是纯 Transformer，而是 convolution 提局部/频段特征，attention 建长程依赖。
5. **Feature + model fusion**：DE、covariance、connectivity、raw EEG、score fusion 结合，但需要严格校准和无泄漏选择。

本仓库当前已经覆盖了 1、4、5 的基础结构，但 2、3 还没有成为正式路线主线。结合 DEP/HC gap，下一阶段最值得做的是 domain adaptation / graph-guided alignment / self-supervised pretraining，而不是继续堆更多同质 backbone。

## 4. 推荐路线排序

### 4.1 立即处理：证据和审计

1. **重审 DGCNN**

`fixed_crop_ea_dgcnn` 已有 25 个 P1 jobs 完成，aggregate metric mean `0.693190`，min `0.623333`，max `0.742671`。当前风险说明称首次 remote audit 只被缺 route summary 阻塞。它可能比当前正式 `CANDIDATE` champion `fixed_crop_ea_fbstcnet` 更稳，但必须先重跑 candidate audit。

命令模板：

```bash
python scripts/repo_doctor.py experiment \
  --route configs/routes/models/fixed_crop_ea_dgcnn.yaml \
  --run outputs/protocol_p1_full_lane_pr15_models/fixed_crop_ea_dgcnn/job_runs/p1__fixed_crop_ea_dgcnn__seed42__fold0 \
  --gate candidate
```

如果 candidate gate 设计不接受单 job run-dir 表达 P1 aggregate，就应先修 summary/audit binding，而不是手工升级状态。

2. **补齐 FBCNet / FBCNet train-aug**

`fixed_crop_ea_fbcnet` 和 `fixed_crop_ea_fbcnet_train_aug` 已有 24/25 partial jobs，均值约 `0.644`，但因 fail-fast 中断，不能决策。补齐后可作为 filterbank variance-layer 对照。

3. **清理 TBD 占位 summary**

`fixed_crop_ea_lggnet`、`fixed_crop_ea_tsception`、`fixed_crop_ea_tsception_train_aug`、`fixed_crop_pure_deformer_lite`、`tuned_sliding_window_*` 等 summary 写了 `TBD`。这种占位可帮助 remote workflow，但最终报告中必须明确标为 `PENDING_REMOTE_P1`，并在正式结果出来后替换，避免被 route_board 误读为已有证据。

### 4.2 主线：FBSTCNet fixed-crop 收敛

优先 P1/P2 的路线：

| 优先级 | route | 理由 |
|---:|---|---|
| 1 | `fixed_crop_car_fbstcnet` | 当前最高 single-split `0.706457`，但 DEP gap 大 |
| 2 | `fixed_crop_ea_whitening_eps1e3_fbstcnet` | 接近最高，whitening 方向有外部合理性 |
| 3 | `fixed_crop_ea_fbstcnet_c_only` | connectivity branch 单独强，能验证脑网络假设 |
| 4 | `fixed_crop_ea_fbstcnet_bands_4_40` | band range 控制，帮助判断 48-52Hz 是否噪声 |
| 5 | `fixed_crop_fbstcnet_zscore_only` | 反向检验 EA 是否总是必要 |

停止或降权：

| route/family | 原因 |
|---|---|
| `fixed_crop_ea_fbstcnet_p_only` | 明显低于 C-only，power-only 不是当前主线 |
| `fixed_crop_ea_fbstcnet_f1_24_f2_48` | capacity 调整没有超过结构/预处理收益 |
| 大量 Deformer 小变体 | 单 split 低于 FBSTCNet champion，且没有直接解决 DEP gap |

### 4.3 第二主线：graph/domain adaptation

建议把 graph 路线从“补充模型”升级为“解决 cohort gap 的路线”：

1. `fixed_crop_ea_dgcnn` 先重审。
2. `fixed_crop_ea_lggnet` 完成 P1，和 DGCNN 比较。
3. 增加 graph-guided domain adaptation 的最小 route，使用 train-side source subjects 估计 alignment，不使用 test statistics。
4. 对所有 graph route 输出 adjacency/region provenance，避免图结构成为不可审计黑盒。

### 4.4 第三主线：self-supervised/foundation，但要先建 provenance

不建议马上大规模引入 LaBraM/CBraMod/REVE 作为黑盒。更稳的最小路线：

1. 把当前 `cbramod` route 明确命名或标注为 scratch/lite。
2. 新增一个 `pretrained_eeg_encoder` 接口，而不是把 checkpoint 加载塞进 backbone。
3. 先做 frozen encoder + linear head smoke，再做 partial fine-tune。
4. manifest 记录 pretraining source、weight hash、channel mapping、sample rate adapter、frozen layers、normalization scope。

如果做不到这些，foundation route 的科学可信度会低于当前 FBSTCNet/graph route。

## 5. 建议的执行计划

### Phase A：证据收敛，不开新路线

目标：让 route board 可读、可审计。

1. 重跑 `fixed_crop_ea_dgcnn` candidate audit。
2. 补齐 `fixed_crop_ea_fbcnet*` 剩余 jobs。
3. 将所有 `TBD` summary 明确标记为 placeholder，正式结果出来后替换。
4. 生成一张 `model_family_board.md`，按 family 而不是 route 平铺。

### Phase B：三条强路线正式 P1/P2

目标：确认 single-split champion 是否稳定。

路线：

```text
fixed_crop_car_fbstcnet
fixed_crop_ea_whitening_eps1e3_fbstcnet
fixed_crop_ea_fbstcnet_c_only
```

每条必须输出：

```text
P1 mean/min/max/std
DEP/HC mean/min/std
crop1-5 BA
random/worst crop BA
hard subject list
score calibration summary
```

### Phase C：fusion 和 domain adaptation

目标：不再用旧 component 做高分堆叠。

1. 用 Phase B/P1 PASS 的 component 重建 fusion。
2. fusion 权重只在 train/val 或 P3 inner fold 选择。
3. 增加 domain adaptation route，目标不是 overall +0.01，而是 DEP/HC gap 收敛。

### Phase D：foundation model proof of concept

目标：验证预训练是否真的比 scratch 有价值。

1. frozen pretrained encoder + linear head。
2. frozen + small MLP head。
3. last-N-layer fine-tune。
4. 与 `fixed_crop_ea_fbstcnet` 和 `fixed_crop_ea_dgcnn` 同协议比较。

## 6. 不建议继续做的事

1. 不建议继续增加 Deformer depth/dropout/embedding 的近邻变体，除非 P1 证明它比 FBSTCNet 更稳。
2. 不建议把 CBraMod scratch route 当作 foundation route 汇报。
3. 不建议在单 split 上继续调 fusion 权重。
4. 不建议只看 overall BA 忽略 DEP/HC gap。
5. 不建议把 `latest_gate: candidate` 的 IDEA route 直接理解为可 promotion route。
6. 不建议为了追分修改 split、crop policy 或 route status。

## 7. 外部参考

数据集与任务：

- [TorchEEG BCI2022Dataset](https://torcheeg.readthedocs.io/en/v1.0.10/generated/torcheeg.datasets.BCI2022Dataset.html)
- [DEAP: A Database for Emotion Analysis Using Physiological Signals](https://research.utwente.nl/en/publications/deap-a-database-for-emotion-analysis-using-physiological-signals/)
- [SEED Dataset, BCMI SJTU](https://bcmi.sjtu.edu.cn/home/seed/)

评估与泛化：

- [Toward cross-subject and cross-session generalization in EEG-based emotion recognition](https://www.sciencedirect.com/science/article/pii/S0925231224011251)
- [Data leakage in deep learning studies of translational EEG](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1373515/full)
- [Transfer Learning for Brain-Computer Interfaces: A Euclidean Space Data Alignment Approach](https://arxiv.org/abs/1808.05464)

经典和仓库相关模型：

- [EEGNet](https://arxiv.org/abs/1611.08024)
- [FBCNet](https://arxiv.org/abs/2104.01233)
- [FBSTCNet, DBLP record](https://dblp.dagstuhl.de/rec/journals/taffco/HuangWLW24.html)
- [TSception](https://arxiv.org/abs/2104.02935)
- [LGGNet](https://arxiv.org/abs/2105.02786)
- [EEG Conformer, PubMed](https://pubmed.ncbi.nlm.nih.gov/37015413/)

domain adaptation / self-supervised / foundation：

- [DANN-MAT for EEG emotion recognition](https://www.sciencedirect.com/science/article/pii/S0957417424028951)
- [MS-DCDA](https://arxiv.org/abs/2408.10235)
- [EGDA graph-guided domain adaptation](https://arxiv.org/abs/2512.23526)
- [EEG-SCMM](https://arxiv.org/abs/2408.09186)
- [Cascaded Self-supervised Learning for Subject-independent EEG Emotion Recognition](https://arxiv.org/abs/2403.04041)
- [BENDR](https://arxiv.org/abs/2101.12037)
- [LaBraM](https://arxiv.org/abs/2405.18765)
- [CBraMod](https://arxiv.org/abs/2412.07236)
- [REVE](https://arxiv.org/abs/2510.21585)
