# 评估协议怎么用

这个仓库保留三种严谨评估协议：P1、P2、P3。  
它们用途不同，不能混成一个分数来解释。

当前实现状态：

```text
已实现：协议配置、协议 plan/dry-run、runner manifest、审计口径说明、component trainer、真实 HUST EEG `.mat` loader、`torch_classifier` candidate job adapter、P2 checkpoint reuse、P3 nested selection + outer-train final retrain。
未实现：已提交绑定 summary 的真实 candidate 结果。
原则：先生成计划和证据边界，再显式启动重训练；不能用 runner manifest 伪装 candidate 结果。
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

materialize runner：

```bash
python scripts/run_evaluation_protocol.py \
  --protocol p1 \
  --route-config configs/routes/models/ea_deformer.yaml \
  --run-dir outputs/protocol_runs/<run_id>
```

runner 会在 `outputs/protocol_runs/<run_id>/splits/` 里写每个 job 的 split contract，并在 `protocol_run_manifest.json` 中记录 `split_manifest_path` 和 `split_sha256`。这些 contract 只是 job-specific 占位锁；真实训练前必须把它们替换为包含 subject lists 和 `trial_rows` 的正式 evidence。

如果给 P1 runner 传入真实 HUST EEG 数据根目录，runner 会直接从 `.mat` 文件索引 subject/trial，并生成正式 subject-level `train_subjects`、`val_subjects`、`test_subjects` 和 `trial_rows`。`--device` 是执行设备契约的一部分：它会写入 `protocol_run_manifest.json`，执行时传给 `run_route_job.py`，最终由 `torch_classifier` 记录为 `requested_device` / `resolved_device`。

```bash
python scripts/run_evaluation_protocol.py \
  --protocol p1 \
  --route-config configs/routes/models/fixed_crop_ea_fbcnet.yaml \
  --run-dir outputs/protocol_runs/p1_fbcnet_<run_id> \
  --data-root scratch/local_data/hust_bci_er_train/训练集 \
  --device cuda \
  --seed 42 \
  --n-folds 5
```

当前 runner 还提供受限执行模式：

```bash
python scripts/run_evaluation_protocol.py \
  --protocol p1 \
  --route-config configs/routes/models/toy_eegnet.yaml \
  --run-dir outputs/protocol_runs/toy_p1 \
  --seed 42 \
  --n-folds 2 \
  --execute \
  --execute-gate smoke \
  --max-execute-jobs 1
```

`--execute` 只调用已支持的 route job adapter。toy route 用于端到端平台 smoke；真实 HUST EEG P1 route 在传入 `--data-root` 后可用于受控 diagnostic/smoke 执行，例如 `--execute-gate smoke --execute-epochs-override 1 --device cuda`。这种运行用于证明真实数据、split、device、manifest、prediction 和 score matrix 链路可跑通；正式 candidate 证据必须使用 route 原始 epochs，不能带 epoch override，并且需要按审计要求补齐 route summary 与 candidate gate。

P2/P3 现在可以生成 plan、runner manifest 和 split contract，也可以执行 artifact-only 训练/选择 job。P2 的 `train_holdout_model` 会写出可复用 `torch.save` checkpoint，后续 holdout crop-policy eval job 会在 artifact 存在时复用同一个 checkpoint；P3 的 `inner_select` 会写出 checkpoint 和 `selection_metrics.json` 作为参数选择证据，outer final job 会读取这些 selection artifacts，按 primary metric 的 inner mean 选择 best `param_index` / `param_overrides`，再用选中参数在 outer train 范围内重新训练 final artifact，并只在 outer test 上评估。P3 final manifest 会写入 `protocol_selected_artifact`，记录选中参数、selection metrics、inner checkpoint provenance 和 final effective route config；inner checkpoint 不会被直接复用到 outer-test final evaluation。`--execute-gate candidate` 不允许 epoch override，也不允许用 `--max-execute-jobs` 跳过未完成 job。

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

P2 crop policy 固定规则：

```text
crop1-5 -> 固定 crop index 0-4
random  -> 每个 trial 用 numpy.default_rng(train_seed) 选择 crop；调用时必须显式传 seed
worst   -> label-aware stress test，选择使 metric 最低的 assignment；tie-break 为 lowest_assignment_index
```

## P3：nested selection

做什么：

```text
outer fold 只负责最终 test
inner fold 只负责选择参数
选好参数后，outer final 在 outer train 范围内重新训练 final artifact
final artifact 只在 outer test 上评估
selected params / selection artifacts 会写进 final manifest
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

`--grid-size` 只用于 `plan_evaluation_protocol.py` 的工作量估算。要 materialize 或 execute 多参数 P3，必须传入实际 search-space YAML：

```bash
python scripts/run_evaluation_protocol.py \
  --protocol p3 \
  --route-config configs/routes/models/sliding_window_eegnet.yaml \
  --param-grid sliding_window_eegnet=configs/search/spaces/sliding_baseline.yaml \
  --run-dir outputs/protocol_runs/p3_sliding_window_eegnet_<run_id>
```

`--param-grid` 使用和 `scripts/hparam_search.py` 相近的 `parameters.<dot.path>.values` / `coarse` / `fine` 形状，但这里只展开一组具体候选列表，不执行 `hparam_search.py` 的 coarse-to-fine 两阶段搜索过程。单个参数 spec 的取值优先级是 `values`，否则 `coarse`，否则 `fine`；如果同一个 search-space 同时有 `coarse` 和 `fine`，P3 只会展开 `coarse`。需要 fine-stage 候选时，应传入只描述 fine 候选的独立 search-space 文件。每个 inner job 会把对应 `param_overrides` 写入 `selection_metrics.json`；outer final 只复用这些参数选择证据，不复用 inner checkpoint 权重。`run_evaluation_protocol.py` 会拒绝 `--grid-size > 1` 但没有 `--param-grid` 的 P3 materialization，避免生成只有抽象 `param_index`、没有真实参数值的 runnable manifest。

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

## 滑动窗口路线家族

`sliding_window_conformer_lite`、`sliding_window_deformer_lite`、`sliding_window_eegnet`、`sliding_window_srfnet` 共享同一套 split-first 滑动窗口基线：

```text
augmentation.name: split_first_sliding_window
source_trial_sec: 10
window_sec: 6
stride_sec: 1
apply_to_splits: [train, val, test]
aggregate_to_trial.method: majority_vote
aggregate_to_trial.tie_break: mean_score
inference.crop_policy: sliding_window_vote
```

这组 route 只应该在 `route_id` 和 `model.name` 上不同；如果窗口长度、stride、聚合方式或 tie-break 变化，应作为整个家族基线变更同步更新四个 route config。

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
更像“带选择过程和 final retrain 的无偏验证”。
重点看 outer-test 结果、选中参数、final retrain 结果和参数选择稳定性。
```

## 当前限制

当前仓库已有统一 runner manifest、toy end-to-end smoke，以及真实 HUST EEG `.mat` loader / `torch_classifier` candidate adapter。
`plan_evaluation_protocol.py` 只生成计划，不训练，不读取标签，不生成结果。  
`run_evaluation_protocol.py` 会生成可审计 job 清单和 lock 信息；P1 在提供真实 `--data-root` 时会生成正式 subject/trial split contract，并可通过 `--execute` 跑受控诊断 job。真实 candidate 结果应按 route 默认训练轮数生成，并由 `repo_doctor.py experiment --gate candidate` 审计。真实数据根目录通过 `--data-root` 或 `HUST_BCI_ER_DATA_ROOT` 提供，`.mat` 文件必须是 HDF5/v7.3 格式。带 `--epochs-override` 的运行只用于链路验证，会被 candidate/promoted gate 阻止。

无 sliding-window augmentation 的 exact-metric route（例如 `ea_deformer`）在 candidate 模式下不是普通单 crop 证据：adapter 会为 held-out test trial 生成 5 个不重叠 fixed crops 写入 `score_matrix.csv`，并要求 dataset manifest 中的 `crop_ids/window_start_secs` 与之对齐。原始 trial 必须至少覆盖 `5 * input_window_sec` 秒；`predictions.csv` 的 trial-level score 是这些 fixed crops 的均值，审计主指标以 score matrix 复算为准。

后续要提交真实 candidate 结论时，必须具备：

```text
正式 dataset manifest
正式 split manifest
held-out test prediction / score matrix
run manifest
candidate audit report
绑定 audit/manifest 的 route summary
```
