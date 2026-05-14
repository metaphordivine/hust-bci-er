# HUST BCI ER

Clean EEG/BCI emotion-recognition research workspace.

This repository is organized around reusable components, route configs, unified evaluation, and auditable experiment records.

Start here:

1. Read `docs/目录怎么用.md`.
2. Read `AGENTS.md`.
3. Validate the skeleton:

```bash
python scripts/check_repo_conventions.py
python scripts/validate_route.py --all
python -m pytest tests -q
```

No trained parameters, checkpoints, raw outputs, public/private labels, or leaderboard feedback should be committed.

All migrated routes are peers by default. Do not use route directories such as
`stable`, `candidates`, `experimental`, or `deprecated` to pre-rank models.
Route names should describe components and evaluation behavior directly rather
than using historical shorthand or performance labels.
