# Experiment Audit Protocol

This protocol is platform-neutral. It can be followed by a human, local AI assistant, CI job, or any automation runner.

The protocol does not make scientific decisions by natural language judgment. It calls repository commands and uses exit codes plus audit artifacts.

## Trigger

Run this when an experiment is said to be finished, when a route is proposed as a candidate, or before merging a PR that changes configs, evaluation, audit code, or route reports.

## Required Inputs

- Route config path.
- Run output directory, if an experiment was executed.
- Expected evaluation protocol.
- Intended route status: `SMOKE_ONLY`, `DIAGNOSTIC_ONLY`, `CANDIDATE`, `PROMOTED`, `REJECTED`, or `ARCHIVED`.

## Required Commands

```bash
python scripts/check_repo_conventions.py
python scripts/validate_route.py --all
python -m pytest tests -q
```

If a run directory exists:

```bash
python scripts/audit_experiment.py --route <route_config> --run <run_dir>
```

## Decision Rules

| Result | Decision |
|---|---|
| leakage, split, or schema guard fails | `REJECT` |
| metrics cannot be recomputed | `DIAGNOSTIC_ONLY` |
| manifest is missing | `DIAGNOSTIC_ONLY` |
| route summary is missing | `BLOCKED` |
| only smoke was run | `SMOKE_ONLY` |
| all required checks pass | route may advance to its requested status |

## Gate Types

| Gate | When | Contents |
|---|---|---|
| Fast gate | every PR | schema, naming, no-leakage tests, metric unit tests |
| Experiment gate | after a local experiment | manifest, prediction shape, metric recompute, summary check |
| Heavy gate | manual or scheduled | full CV, pseudo-public holdout, worst crop stress test |

