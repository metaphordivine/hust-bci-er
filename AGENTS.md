# Agent Rules

These rules apply to all local collaborators, AI assistants, automation agents, and CI jobs. They are platform-neutral and do not depend on a specific assistant product.

## Required First Reads

Before code, experiment, or report changes:

1. Read `README.md`.
2. Read `docs/目录怎么用.md`.
3. Read `docs/04_evaluation_protocols.md`.
4. Read `agent_protocols/experiment_audit.md`.
5. Read `agent_protocols/skill_router.md`.
6. Read this file.

## Skill Routing

Humans do not need to name the exact repository skill. If a request describes a scenario, the agent must choose the matching skill from `agent_protocols/skill_router.md`, state the selected skill briefly, and follow its required commands and completion gate.

If multiple skills are required, state the order before editing. Use the narrowest skill that satisfies the task and do not touch unrelated files.

Examples:

- "run repo health check" -> Repository Fast Gate.
- "can this run become candidate?" -> Experiment Audit Skill.
- "compare P1/P2/P3 workload" -> Evaluation Protocol Planning Skill.
- "assemble component scores" -> Score Route Assembly Skill.
- "review this PR" -> Chinese Code Review Skill.

## Allowed

- Add route configs under `configs/routes/models/`.
- Add reusable components under `src/hust_bci_er/`.
- Add tests for new components.
- Add concise route summaries under `reports/route_summaries/`.
- Add local-only experimental files under `outputs/` or `scratch/`.

## Forbidden

- Do not commit checkpoints, raw outputs, full prediction tables, or temporary experiment dumps.
- Do not use public/private labels or leaderboard feedback.
- Do not use `subject_id`, `trial_id`, `pseudo_trial_id`, or filenames as model features.
- Do not create one-off giant scripts in `scripts/`.
- Do not silently change split definitions or evaluation protocols.

## Code Review Language

- All automated code review summaries and review comments for this repository must be written in Simplified Chinese.
- Risk levels, issue descriptions, suggested fixes, and final review conclusions must be written in Simplified Chinese.
- Keep code identifiers, file paths, function names, class names, commands, and error messages in their original spelling.
- Do not translate variable names, API names, config keys, or command output literally into Chinese.

## Required Before Completion

Run:

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

If any command fails, report the failure and keep the route status unchanged unless the audit report explicitly supports a status change.
