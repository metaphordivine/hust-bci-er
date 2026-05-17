# Agent Skill Router

This protocol is platform-neutral. It tells any human collaborator, local AI
assistant, CI job, or automation runner how to route a task to the right
repository skill and minimal context pack.

Humans do not need to name the exact skill. If the request describes a scenario,
the agent must select the matching skill, classify intake mode, read the
selected context pack, run the expected commands, and report the result in
Simplified Chinese.

## Routing Table

| Human scenario | Repository skill | Intake mode | Context pack | Required first action | Completion gate |
|---|---|---|---|---|---|
| "check the repo", "before PR", "make sure it is clean" | Repository Fast Gate | `command-only` | `agent_context/repo_fast_pack.md` | Run `python scripts/repo_doctor.py fast` | Command exits 0 |
| "add a route", "change route config" | Route Config Skill | `direct-task` | `agent_context/route_config_pack.md` | Select route-config pack and edit only route config scope unless more scope is explicit | `python scripts/validate_route.py --all` and fast gate pass |
| "which evaluation should I use", "plan P1/P2/P3" | Evaluation Protocol Planning Skill | `planning-only` | `agent_context/protocol_runner_pack.md` | Run or inspect `scripts/plan_evaluation_protocol.py` for requested protocol(s) | Plan output explains jobs, use case, and guardrails |
| "assemble this score route", "assemble component scores", "combine component scores" | Score Route Assembly Skill | `direct-task` | `agent_context/score_fusion_pack.md` | Use score-fusion pack and `scripts/assemble_score_route.py` / score assembly code | Output prediction table has `route_id`, `subject_id`, `trial_id`, `score`, `pred_top4` |
| "experiment is finished", "can this be candidate" | Experiment Audit Skill | `state-changing` | `agent_context/evidence_gate_pack.md` | Run `repo_doctor.py experiment --route ... --run ... --gate candidate`; do not edit status without PASS evidence | Candidate gate exits 0; otherwise keep route status unchanged |
| "just smoke this route" | Smoke Audit Skill | `command-only` | `agent_context/evidence_gate_pack.md` | Run `repo_doctor.py experiment --route ... --gate smoke` | WARN is allowed; do not call it candidate evidence |
| "changed model code", "changed Torch backbone" | Model Smoke Skill | `direct-task` | `agent_context/model_backbone_pack.md` | Read model-backbone pack and run targeted model smoke/tests | All registered Torch backbones forward successfully |
| "changed dataset or split evidence" | Evidence and Leakage Guard Skill | `state-changing` | `agent_context/evidence_gate_pack.md` | Validate dataset/split manifests and run fast gate; ask before split changes without explicit approval | No subject overlap, no original-trial split leakage, fast gate exits 0 |
| "use the common foundations", "make the agent use the base modules", "wire reusable foundation artifacts" | Foundation Usage Skill | `direct-task` | `agent_context/evidence_gate_pack.md` | Read `agent_protocols/foundation_usage.md` only if selected pack or task requires it, then select matching foundation row | Narrow artifact validation and `python scripts/repo_doctor.py fast` pass |
| "build dataset manifest", "build split manifest", "formal evidence", "dataset QA" | Foundation Evidence Skill | `state-changing` | `agent_context/evidence_gate_pack.md` | Use dataset/split builders and QA commands from `agent_protocols/foundation_usage.md` | Dataset/split evidence is auditable and fast gate passes |
| "write predictions", "write metric report", "write route summary", "generate route summary", "write run manifest", "combine artifact reports" | Foundation Artifact Skill | `direct-task` | `agent_context/summary_report_pack.md` | Use contracts, prediction writer, metric report builder, route summary generator, score route assembly outputs, and run manifest writer | Artifact contract validates; experiment gate passes when run artifacts exist |
| "track training", "cache derived features", "refresh component docs", "check registry consistency" | Foundation Maintenance Skill | `direct-task` | `agent_context/repo_fast_pack.md` | Use training monitor, cache manager, registry checker, or docs generator from `agent_protocols/foundation_usage.md` | Generated maintenance artifact is local or concise, and fast gate passes |
| "prepare promoted evidence" | Promotion Audit Skill | `state-changing` | `agent_context/evidence_gate_pack.md` | Fill `reports/promotion_audits/<route_id>_promotion.md` from template only with passing candidate audit evidence | Promoted gate references a passing candidate audit report |
| "review this PR/code" | Chinese Code Review Skill | `planning-only` | `agent_context/review_pack.md` | Inspect changes and lead with findings | All review summaries and comments are in Simplified Chinese |
| "fix PR review", "修 PR17 review", "conversation review" | Review Fix Skill | `review-fix` | `agent_context/review_fix_pack.md` | Fetch or ingest reviews with `scripts/agent_review_inbox.py`; create issue board before implementation edits | Targeted tests and fast gate pass |
| "执行下面计划", long AI review, S0/S1/S2 plan | Plan Ingest Skill | `plan-driven` | task-specific pack selected by `scripts/agent_intake.py` | Save raw plan, run `scripts/agent_plan_ingest.py`, create issue board before implementation edits | Issue board items fixed/deferred and validation passes |
| "agent intake", "context pack", "minimal context", "context-engineering" | Context Engineering Skill | `direct-task` | `agent_context/context_engineering_pack.md` | Update agent-facing context/intake/session docs and tests only | Intake/context tests and fast gate pass |

## Required Intake Documents

Every agent must read these before making code, experiment, Git, or report
changes:

1. `agent_protocols/AGENT_BRIEF.md`
2. `agent_protocols/skill_router.md`

Then classify the request with `scripts/agent_intake.py`, select a context pack
with `scripts/agent_context.py`, and read only that pack's listed files. Intake
is DeepSeek-first when `DEEPSEEK_API_KEY` is available and falls back to
deterministic heuristics when the key or API is unavailable. Full protocol docs
are read only when the selected pack requires them, a validation failure points
there, or the task explicitly edits that protocol.

## Skill Rules

- Choose the narrowest skill that satisfies the task.
- If multiple skills are needed, state the order before editing.
- Do not ask the human to choose a skill when the scenario is clear.
- If the human provides a long operation plan, PR review, or AI review, route to `plan-driven` or `review-fix`.
- The agent must create an issue board before editing implementation files in `plan-driven` or `review-fix` mode.
- If no obvious plan is present but the task is clear, use `direct-task` with a micro-plan.
- Do not silently upgrade smoke or diagnostic output into candidate evidence.
- Do not change route status unless an audit report supports that status.
- Do not commit `outputs/`, checkpoints, full prediction tables, or temporary dumps.
- Do not touch unrelated directories when a skill has a narrow file scope.
- Run `python scripts/repo_doctor.py fast` before final handoff unless the task is read-only or explicitly planning-only.

## Intake Examples

```text
执行下面计划
```

Use `plan-driven`, run `scripts/agent_plan_ingest.py`, create an issue board,
then read the selected context pack.

```text
修 PR17 review
```

Use `review-fix`, run `scripts/agent_review_inbox.py` when a PR number is
available, create an issue board, then implement selected fixes.

```text
score-fusion：修 whitening 绑定
```

Use `direct-task` with `agent_context/score_fusion_pack.md` and a short
session micro-plan.

```text
现在该怎么做
```

Use `planning-only`; do bounded triage and output a plan without code edits.

```text
跑 fast gate
```

Use `command-only`; run the requested command and report.

## Parallel Human Work With Agent Assistance

Multiple human collaborators may work in parallel with their own local AI
assistants when their write scopes do not overlap. The human collaborator
remains responsible for scope, validation, and final handoff.

Avoid parallel edits to shared governance files unless a human owner explicitly
coordinates the change. `reports/route_registry.yaml` lists the current shared
guarded paths and route owners.

Common guarded paths:

```text
README.md
AGENTS.md
agent_protocols/
agent_context/
docs/04_evaluation_protocols.md
src/hust_bci_er/config/schema.py
src/hust_bci_er/config/registry.py
scripts/repo_doctor.py
.github/workflows/
reports/route_board.md
```

## Human Prompts That Are Enough

Humans can give short task prompts. The agent should infer the skill and intake
mode.

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
帮 agent 用公共基座把 dataset/split/run artifact 串起来。
```

Route to Foundation Usage Skill, then select the needed foundation rows.

```text
中文 review 这个 PR。
```

Route to Chinese Code Review Skill.
