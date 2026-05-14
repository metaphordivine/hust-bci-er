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

## Required Before Completion

Run:

```bash
python scripts/check_repo_conventions.py
python scripts/validate_route.py --all
python -m pytest tests -q
```

For a finished experiment run:

```bash
python scripts/audit_experiment.py --route <route_config> --run <run_dir>
```

If any command fails, report the failure and keep the route blocked or diagnostic-only.
