# Agent Rules

These rules apply to all local collaborators, AI assistants, automation agents, and CI jobs. They are platform-neutral and do not depend on a specific assistant product.

## Required First Reads

Before code, experiment, or report changes:

1. Read `README.md`.
2. Read `docs/目录怎么用.md`.
3. Read `agent_protocols/experiment_audit.md`.
4. Read this file.

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
