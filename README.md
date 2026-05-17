# HUST BCI ER

干净、可复用、可审计的 EEG/BCI 情绪识别研究工作区。

这个仓库围绕四件事组织：

```text
可复用组件 + route config + 统一评估口径 + 可追溯实验审计
```

## 快速开始

agent 协作默认使用最小上下文。先读：

1. `agent_protocols/AGENT_BRIEF.md`
2. `agent_protocols/skill_router.md`
3. 用 `scripts/agent_intake.py` 分类任务
4. 用 `scripts/agent_context.py --task <task_family>` 选择最小 context pack

`scripts/agent_intake.py` 默认使用本地 deterministic heuristics。只有显式加
`--deepseek`，或设置 `AGENT_INTAKE_ENGINE=deepseek` 时，才会调用 DeepSeek；
API 不可用时会回退本地规则，除非同时加 `--require-deepseek`。

只有 selected context pack、验证失败或任务本身要求时，才继续读完整协议文档，例如 `docs/04_evaluation_protocols.md` 或 `agent_protocols/foundation_usage.md`。

然后运行仓库 fast gate：

```bash
python scripts/repo_doctor.py fast
```

人类不需要记住具体 skill 名。直接描述任务场景即可，agent 必须按 `agent_protocols/skill_router.md` 自动选择对应 skill、运行命令并用中文汇报结果。

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
- PyTorch classifier component trainer。
- deterministic runtime helpers、environment/requirements lock、run manifest writer。
- P1/P2/P3 protocol runner manifest materialization，以及 P2/P3 checkpoint/selection artifact 执行与复用。
- score route 从 component score CSV 到 `score/pred_top4` prediction table 的最小执行入口。
- Foundation Usage Skill，用于把 dataset/split evidence、prediction/report、run manifest、promotion、registry/cache/monitor 等公共基座路由成稳定 agent 工作流。
- toy end-to-end audit smoke：`toy_eegnet` 可生成 synthetic dataset/split、prediction、score matrix、metric report、run manifest，并通过 candidate audit，用于 CI 和新人环境验证。
- 真实 HUST EEG `.mat` dataset loader 和 `torch_classifier` job adapter；支持 `smoke`、`full_subjects` 诊断模式和 held-out test `candidate` 模式。
- route registry / route board：`reports/route_registry.yaml` 记录 route owner/blocker 和共享谨慎修改路径，`reports/route_board.md` 从 route config 与 registry 生成。
- LGGNet/DGCNN-style 图模型、TSception/FBCNet-style EEG backbone、通用 linear/MLP classifier head、Butterworth bandpass preprocessing、connectivity / Riemannian tangent features，以及 train-only augmentation transforms。

仍属于预留或后续实现：

- 复杂训练 callback。
- 按 route 默认训练轮数完成并提交绑定 summary 的 candidate 级真实实验结果。

P1/P2/P3 当前已迁入为 protocol 配置、dry-run plan 和统一 runner manifest。plan 只看工作量；runner 会锁定 route/config/dataset/source split、job-specific split contract、seed、environment 和 job artifact contract。P2/P3 的 artifact-only job 会写出可复用 checkpoint / selection artifact；candidate 执行不会跳过这些依赖，也不会伪造训练结果：

```bash
python scripts/plan_evaluation_protocol.py --protocol p1 --route-config configs/routes/models/ea_deformer.yaml
python scripts/plan_evaluation_protocol.py --protocol p2 --route-config configs/routes/models/ea_deformer.yaml
python scripts/plan_evaluation_protocol.py --protocol p3 --route-config configs/routes/models/ea_deformer.yaml --grid-size ea_deformer=6
python scripts/run_evaluation_protocol.py --protocol p2 --route-config configs/routes/models/ea_deformer.yaml --run-dir outputs/protocol_runs/<run_id>
```

协议 runner 可用 `--execute` 驱动当前支持的 job adapter。toy route 用于 CI 级平台 smoke：

```bash
python scripts/run_evaluation_protocol.py --protocol p1 --route-config configs/routes/models/toy_eegnet.yaml --run-dir outputs/protocol_runs/toy_p1 --seed 42 --n-folds 2 --execute --execute-gate smoke --max-execute-jobs 1
```

Torch 模型 forward smoke 不在默认 fast gate 中运行；改动模型路径时由 `.github/workflows/model_smoke.yml` 触发，也可以手动运行：

```bash
pip install -e ".[dev,models]"
python scripts/model_smoke.py
```

真实 HUST EEG route 可通过 `run_candidate_route.py` 运行。`.mat` 文件需要是 `h5py` 可读取的 HDF5/v7.3 MATLAB 文件；本机数据根目录可用 `--data-root` 显式传入，也可设置 `HUST_BCI_ER_DATA_ROOT`。`candidate` 模式使用全部可用 subject，训练集只用于拟合，val 用于 checkpoint selection，prediction/score matrix 只写 held-out test subjects；脚本会在唯一阻塞项是缺少 route summary 时生成绑定 audit/manifest 的 summary 并重跑 candidate gate：

```bash
python scripts/launch_reproducible.py --seed 42 -- \
  python scripts/run_candidate_route.py \
    --route configs/routes/models/sliding_window_eegnet.yaml \
    --run-dir outputs/sliding_window_eegnet/<run_id> \
    --data-root scratch/local_data/hust_bci_er_train/训练集
```

P1 repeated group-kfold 的训练前批量入口是 `run_evaluation_protocol.py`。传入真实 `--data-root` 时，runner 会索引 HUST `.mat` 文件并为每个 seed/fold 写入正式 subject-level split contract 和 `trial_rows`；`--device` 会记录到 protocol manifest，并在 `--execute` 时传给每个 route job 与最终训练 manifest。诊断跑可以加 `--execute-epochs-override`，但它不能和 `--execute-gate candidate` 同用：

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

`smoke` 和 `full_subjects` 都是诊断模式，不能作为 candidate 证据。
带 `--epochs-override` 的 candidate run 只适合验证链路，candidate/promoted gate 会阻止它作为正式证据。
对 `ea_deformer` 这类无 sliding-window augmentation、但主指标为 `exact_single_crop_expected_BA` 的真实 candidate route，adapter 会为 score matrix evidence 从每个 held-out test trial 切出 5 个不重叠固定 crop。因此原始 trial 至少需要 `5 * input_window_sec` 秒；`ea_deformer` 的 `input_window_sec: 10` 对应每个 trial 至少 50 秒。此时 `predictions.csv` 的 trial-level `y_score` 是这 5 个 fixed crops 的 `mean_score` 聚合，主指标仍由 `score_matrix.csv` 复算。

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
candidate gate 还会要求 manifest 记录 environment、determinism、checkpoint_selection、crop_policy 和 environment_lock，并复核 seed/deterministic/crop/checkpoint/lock hash 语义；缺失或不一致时保持 route status 不变。

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
