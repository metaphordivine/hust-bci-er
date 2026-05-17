# Agent Rules

These rules apply to all local collaborators, AI assistants, automation agents,
and CI jobs. They are platform-neutral and do not depend on a specific assistant
product.

## Required Intake

For every task:

1. Read `agent_protocols/AGENT_BRIEF.md`.
2. Read `agent_protocols/skill_router.md`.
3. Classify the user input using `scripts/agent_intake.py`.
4. Run or follow `scripts/agent_context.py --task <task_family>`.
5. Read only the selected context pack and listed files.
6. Read full protocol docs only if the selected pack requires them, validation failure points there, or the task explicitly edits that protocol.

Minimal context is the default. Humans do not need to say "use minimal context".
When `DEEPSEEK_API_KEY` is available, `scripts/agent_intake.py` uses
DeepSeek-first intake and falls back to deterministic heuristics if the API is
unavailable.

`plan-driven` mode is used only when intake detects a clear plan, long AI/human
review, S0/S1/S2 list, or multi-phase operation plan. Direct tasks do not
require an issue board unless the selected pack says so.

If the input is ambiguous and state-changing, ask before editing. If the input
is ambiguous but read-only or planning-oriented, do bounded triage and report a
plan.

## Skill Routing

Humans do not need to name the exact repository skill. If a request describes a
scenario, the agent must choose the matching skill from
`agent_protocols/skill_router.md`, state the selected skill briefly, and follow
its intake mode, context pack, required commands, and completion gate.

If multiple skills are required, state the order before editing. Use the
narrowest skill that satisfies the task and do not touch unrelated files.

Examples:

- "run repo health check" -> Repository Fast Gate.
- "can this run become candidate?" -> Experiment Audit Skill.
- "compare P1/P2/P3 workload" -> Evaluation Protocol Planning Skill.
- "assemble component scores" -> Score Route Assembly Skill.
- "use the common foundations" -> Foundation Usage Skill (`agent_protocols/foundation_usage.md` when selected context requires it).
- "review this PR" -> Chinese Code Review Skill.
- "修 PR17 review" -> review-fix intake with issue board before edits.

## Allowed

- Add route configs under `configs/routes/models/`.
- Add reusable components under `src/hust_bci_er/`.
- Add tests for new components.
- Add concise route summaries under `reports/route_summaries/`.
- Add agent-facing context, intake, issue-board, and session workflow files.
- Add local-only experimental or session files under `outputs/` or `scratch/`.

## Forbidden

- Do not commit checkpoints, raw outputs, full prediction tables, or temporary experiment dumps.
- Do not use public/private labels or leaderboard feedback.
- Do not use `subject_id`, `trial_id`, `pseudo_trial_id`, or filenames as model features.
- Do not create one-off giant scripts in `scripts/`.
- Do not silently change split definitions or evaluation protocols.
- Do not change route status unless an audit report supports it.

## Code Review Language

- All automated code review summaries and review comments for this repository must be written in Simplified Chinese.
- Risk levels, issue descriptions, suggested fixes, and final review conclusions must be written in Simplified Chinese.
- Keep code identifiers, file paths, function names, class names, commands, and error messages in their original spelling.
- Do not translate variable names, API names, config keys, or command output literally into Chinese.

## Required Before Completion

Run targeted tests first.

Run this before final handoff unless the task is read-only or explicitly
planning-only:

```bash
python scripts/repo_doctor.py fast
```

For a finished experiment run:

```bash
python scripts/repo_doctor.py experiment --route <route_config> --run <run_dir> --gate candidate
```

For a route-only smoke audit, `--run` may be omitted:

```bash
python scripts/repo_doctor.py experiment --route <route_config> --gate smoke
```

If any command fails, report the failure and keep the route status unchanged
unless the audit report explicitly supports a status change.
