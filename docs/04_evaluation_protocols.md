# 评估协议怎么用

这个仓库保留三种严谨评估协议：P1、P2、P3。  
它们用途不同，不能混成一个分数来解释。

当前实现状态：

```text
已实现：协议配置、协议 plan/dry-run、审计口径说明。
未实现：完整训练 runner。
原则：先生成计划和证据边界，再显式启动重训练。
```

## 总览

| 协议 | 用途 | 成本 | 适合回答的问题 |
|---|---|---:|---|
| P1 repeated group k-fold | 严格重训比较 | 高 | 这条路线平均水平和 seed/fold 稳定性如何？ |
| P2 pseudo-public holdout | subject holdout 和 crop policy 压力测试 | 中 | 模型对未知 subject、单 crop、worst crop 是否稳？ |
| P3 nested selection | 带参数选择的最终验证 | 很高 | 调参路线是否在外层 test 上仍然成立？ |

统一红线：

```text
不使用 public/private labels
不使用 leaderboard feedback
不生成 final submission
不修改 split
subject_id / trial_id 只能用于分组、对齐、Top-4、审计
```

## P1：repeated group k-fold

做什么：

```text
多个 seed × 多个 subject-group fold
每个 fold 从零训练
每个 held-out subject 用 exact_single_crop_expected_BA 评估
```

什么时候用：

- 比较两个路线的平均性能。
- 看 seed / fold 波动。
- 给 candidate 提供主要 CV 证据。

不要用来做什么：

- 不适合快速探索。
- 不适合一边看结果一边调参数后继续声称是独立验证。

默认口径：

```text
seeds: [42, 123, 456, 789, 1024]
n_folds: 5
primary_metric: exact_single_crop_expected_BA
secondary_metrics: exact_assignment_all_correct_rate, fold_std, subject_std
```

dry-run：

```bash
python scripts/plan_evaluation_protocol.py \
  --protocol p1 \
  --route-config configs/routes/models/ea_deformer.yaml
```

## P2：pseudo-public holdout

做什么：

```text
先按 subject 划出一组 holdout subjects
只用非 holdout subjects 训练
在 holdout subjects 上模拟单 crop 推理
分别看 crop1-5、random、worst
```

什么时候用：

- 检查模型是否对 crop 位置敏感。
- 检查 worst crop 下是否崩。
- 模拟真实未知 subject 的压力测试。
- 对比 Top-4 和 no-Top4 行为。

不要用来做什么：

- 不应单独作为最终 candidate 主证据。
- 不应根据 holdout 结果反复调到最好后再把同一 holdout 当独立测试。

默认口径：

```text
n_holdout_subjects: 12
holdout_seed: 999
train_seed: 42
crop_policies: [crop1, crop2, crop3, crop4, crop5, random, worst]
primary_metric: exact_single_crop_expected_BA
```

dry-run：

```bash
python scripts/plan_evaluation_protocol.py \
  --protocol p2 \
  --route-config configs/routes/models/ea_deformer.yaml
```

## P3：nested selection

做什么：

```text
outer fold 只负责最终 test
inner fold 只负责选择参数
选好参数后，在 outer train 上重训
最后只在 outer test 上评估
```

什么时候用：

- 路线包含可选参数、权重、dropout、融合参数等。
- 想证明“参数选择过程本身”没有泄露 outer test。
- 最终 scientific review 前做最严格确认。

不要用来做什么：

- 不适合日常快速探索。
- 不适合 GPU 资源紧张时先跑全量。
- 不允许 outer test 参与参数选择。

默认口径：

```text
outer_folds: 5
inner_folds: 3
outer_seed: 42
inner_seed: 123
primary_metric: exact_single_crop_expected_BA
secondary_metrics: exact_assignment_all_correct_rate, param_selection_stability, fold_std
```

dry-run：

```bash
python scripts/plan_evaluation_protocol.py \
  --protocol p3 \
  --route-config configs/routes/models/ea_deformer.yaml \
  --grid-size ea_deformer=6
```

## 怎么选择协议

先做：

```text
P2 或小规模 P1 smoke
```

用于排除明显不稳、crop 敏感、subject holdout 崩溃的路线。

进入主要比较：

```text
P1
```

看平均表现、fold_std、subject_std、all-correct rate。

涉及调参或路线选择：

```text
P3
```

P3 是最贵的，不是默认入口。

## 输出应该怎么解释

P1：

```text
更像“路线平均能力”。
重点看 mean、seed_std、fold_std、subject_std。
```

P2：

```text
更像“真实推理压力测试”。
重点看 normal crop 均值、worst crop、random crop、no-Top4 对照。
```

P3：

```text
更像“带选择过程的无偏验证”。
重点看 outer-test 结果和参数选择稳定性。
```

## 当前限制

当前仓库还没有完整训练 runner。  
`plan_evaluation_protocol.py` 只生成计划，不训练，不读取标签，不生成结果。

后续要真正运行训练时，必须先补：

```text
正式 dataset manifest
正式 split manifest
component trainer
protocol runner
run manifest
candidate audit report
```
