# HUST BCI ER

干净、可复用、可审计的 EEG/BCI 情绪识别研究工作区。

这个仓库围绕四件事组织：

```text
可复用组件 + route config + 统一评估口径 + 可追溯实验审计
```

## 快速开始

先读：

1. `docs/目录怎么用.md`
2. `AGENTS.md`
3. `agent_protocols/experiment_audit.md`

然后运行仓库 fast gate：

```bash
python scripts/repo_doctor.py fast
```

## 基本原则

- 所有模型路线默认平等，不预设优先级。
- route 名称只描述组件、预处理、推理和评估行为。
- 不提交训练参数、checkpoint、raw outputs、完整 prediction table 或临时实验 dump。
- 不使用 public/private labels、leaderboard feedback 或 ID-derived shortcut。
- 不静默修改 split、评估协议或 route 生命周期状态。

## 实验审计

route-only smoke 可以不提供 run directory：

```bash
python scripts/repo_doctor.py experiment --route configs/routes/models/ea_deformer.yaml --gate smoke
```

完成一次真实实验后，进入 candidate 前必须提供 run artifacts：

```bash
python scripts/repo_doctor.py experiment --route <route_config> --run <run_dir> --gate candidate
```

candidate gate 会检查 manifest、prediction CSV、Top-4 语义、主指标复算、split/dataset 证据、summary 绑定证据。

promoted gate 还需要 `reports/promotion_audits/<route_id>_promotion.md`，并引用一个通过的 candidate audit report。

## 结果记录

重要路线只提交简短 summary：

```text
reports/route_summaries/<route_id>_summary.md
```

完整训练输出、模型文件和 trial 级表格留在本地 `outputs/`，不进入主线。
