# Agent Skill Router

This protocol is platform-neutral. It tells any human collaborator, local AI assistant, CI job, or automation runner how to route a task to the right repository skill.

Humans do not need to name the exact skill. If the request describes a scenario, the agent must select the matching skill, read the required documents, run the expected commands, and report the result in Simplified Chinese.

## Routing Table

| Human scenario | Repository skill | Required first action | Completion gate |
|---|---|---|---|
| "check the repo", "before PR", "make sure it is clean" | Repository Fast Gate | Run `python scripts/repo_doctor.py fast` | Command exits 0 |
| "add a route", "change route config" | Route Config Skill | Edit only `configs/routes/models/` unless more scope is explicit | `python scripts/validate_route.py --all` and fast gate pass |
| "which evaluation should I use", "plan P1/P2/P3" | Evaluation Protocol Planning Skill | Run `scripts/plan_evaluation_protocol.py` for the requested protocol(s) | Plan output explains jobs, use case, and guardrails |
| "assemble this score route", "combine component scores" | Score Route Assembly Skill | Use `scripts/assemble_score_route.py` | Output prediction table has `route_id`, `subject_id`, `trial_id`, `score`, `pred_top4` |
| "experiment is finished", "can this be candidate" | Experiment Audit Skill | Run `repo_doctor.py experiment --route ... --run ... --gate candidate` | Candidate gate exits 0; otherwise keep route status unchanged |
| "just smoke this route" | Smoke Audit Skill | Run `repo_doctor.py experiment --route ... --gate smoke` | WARN is allowed; do not call it candidate evidence |
| "changed model code", "changed Torch backbone" | Model Smoke Skill | Run `python scripts/model_smoke.py` in a models-capable environment | All registered Torch backbones forward successfully |
| "changed dataset or split evidence" | Evidence and Leakage Guard Skill | Validate dataset/split manifests and run fast gate | No subject overlap, no original-trial split leakage, fast gate exits 0 |
| "prepare promoted evidence" | Promotion Audit Skill | Fill `reports/promotion_audits/<route_id>_promotion.md` from template | Promoted gate references a passing candidate audit report |
| "review this PR/code" | Chinese Code Review Skill | Inspect changes and lead with findings | All review summaries and comments are in Simplified Chinese |

## Required Documents

Every agent must read these before making code, experiment, Git, or report changes:

1. `README.md`
2. `docs/目录怎么用.md`
3. `docs/04_evaluation_protocols.md`
4. `agent_protocols/experiment_audit.md`
5. `agent_protocols/skill_router.md`
6. `AGENTS.md`

## Skill Rules

- Choose the narrowest skill that satisfies the task.
- If multiple skills are needed, state the order before editing.
- Do not ask the human to choose a skill when the scenario is clear.
- Do not silently upgrade smoke or diagnostic output into candidate evidence.
- Do not change route status unless an audit report supports that status.
- Do not commit `outputs/`, checkpoints, full prediction tables, or temporary dumps.
- Do not touch unrelated directories when a skill has a narrow file scope.
- Run `python scripts/repo_doctor.py fast` before final handoff unless the task is read-only.

## Parallel Agent Work

Two agents can work in parallel only when their write scopes do not overlap.

Safe examples:

```text
Agent A: configs/datasets/, configs/splits/, src/hust_bci_er/data/
Agent B: src/hust_bci_er/evaluation/protocols/, scripts/plan_evaluation_protocol.py
```

```text
Agent A: src/hust_bci_er/inference/, scripts/assemble_score_route.py
Agent B: src/hust_bci_er/training/, scripts/model_smoke.py
```

Avoid parallel edits to shared governance files unless one agent is explicitly assigned as the integrator:

```text
README.md
AGENTS.md
docs/目录怎么用.md
src/hust_bci_er/config/schema.py
src/hust_bci_er/config/registry.py
scripts/repo_doctor.py
.github/workflows/*
```

## Human Prompts That Are Enough

Humans can give short task prompts. The agent should infer the skill.

```text
跑一下仓库健康检查。
```

Route to Repository Fast Gate.

```text
这个 run 能不能进 candidate？
```

Route to Experiment Audit Skill.

```text
帮我比较 P1/P2/P3 的工作量。
```

Route to Evaluation Protocol Planning Skill.

```text
这两个 component score 组装一下。
```

Route to Score Route Assembly Skill.

```text
中文 review 这个 PR。
```

Route to Chinese Code Review Skill.
