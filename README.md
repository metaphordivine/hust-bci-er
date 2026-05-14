# HUST BCI ER

干净、可复用、可审计的 EEG/BCI 情绪识别研究工作区。

这个仓库围绕四件事组织：

```text
可复用组件 + route config + 统一评估口径 + 可追溯实验审计
```

## 快速开始

先读：

1. `docs/目录怎么用.md`
2. `docs/04_evaluation_protocols.md`
3. `AGENTS.md`
4. `agent_protocols/experiment_audit.md`

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

## 当前实现边界

已实现并由 fast gate 覆盖：

- route schema、registry、route board、summary consistency。
- manifest / prediction / Top-4 / primary metric 的审计逻辑。
- 基础预处理、手工特征、Top-4、score route 组装函数。
- score route 从 component score CSV 到 `score/pred_top4` prediction table 的最小执行入口。

仍属于预留或后续实现：

- 完整训练 loop。
- 完整 P1/P2/P3 protocol runner。
- 图模型、heads、复杂训练 callback。
- candidate 级真实实验结果。

P1/P2/P3 当前已迁入为 protocol 配置和 dry-run plan。它们定义评估口径和工作量，不会自动训练：

```bash
python scripts/plan_evaluation_protocol.py --protocol p1 --route-config configs/routes/models/ea_deformer.yaml
python scripts/plan_evaluation_protocol.py --protocol p2 --route-config configs/routes/models/ea_deformer.yaml
python scripts/plan_evaluation_protocol.py --protocol p3 --route-config configs/routes/models/ea_deformer.yaml --grid-size ea_deformer=6
```

Torch 模型 forward smoke 不在默认 fast gate 中运行；改动模型路径时由 `.github/workflows/model_smoke.yml` 触发，也可以手动运行：

```bash
pip install -e ".[dev,models]"
python scripts/model_smoke.py
```

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

score route 可用这个入口从 component score CSV 组装 prediction table：

```bash
python scripts/assemble_score_route.py \
  --route configs/routes/models/conformer_srfnet_score_average.yaml \
  --component-score conformer_component=<conformer_scores.csv> \
  --component-score srfnet_long_component=<srfnet_scores.csv> \
  --output outputs/<route_id>/<run_id>/predictions.csv
```

## 结果记录

重要路线只提交简短 summary：

```text
reports/route_summaries/<route_id>_summary.md
```

完整训练输出、模型文件和 trial 级表格留在本地 `outputs/`，不进入主线。
