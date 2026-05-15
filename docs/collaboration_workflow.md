# Human Collaboration Workflow

This repository is for human-led experiment work with optional agent assistance.
Agents help read protocols, draft code, run gates, generate summaries, and review diffs; humans keep responsibility for route decisions and merge decisions.

## Lightweight Ownership

Use `reports/route_registry.yaml` to record:

- the current owner for each `route_id`;
- the current blocker, if known;
- shared guarded paths that should not be edited casually or in parallel.

`owner: unassigned` is allowed while a route is only an idea. Before a route is proposed for candidate or promoted evidence, set a human owner.

## Scope Notes

Before a PR, the human collaborator should know:

- which route or component they are changing;
- which paths are intentionally in scope;
- which shared guarded paths they avoided or intentionally changed;
- which gate was run.

This is a coordination note, not a permission system. If a guarded path really needs a change, change it in the PR and call it out for review.

## Agent Assistance

If an agent was used, record the broad use in the PR template:

```text
code draft / audit / review / docs
```

Do not treat agent output as evidence by itself. Evidence comes from repository commands such as:

```bash
python scripts/repo_doctor.py fast
python scripts/repo_doctor.py experiment --route <route_config> --run <run_dir> --gate candidate
```

## Shared Guarded Paths

The canonical list lives in `reports/route_registry.yaml`. These paths usually affect everyone:

```text
README.md
AGENTS.md
agent_protocols/
docs/04_evaluation_protocols.md
src/hust_bci_er/config/schema.py
src/hust_bci_er/config/registry.py
scripts/repo_doctor.py
.github/workflows/
reports/route_board.md
```

When these change, the PR should explain why and include `python scripts/repo_doctor.py fast` output.

## Onboarding Smoke

A new collaborator can validate the local setup without touching real EEG data:

```bash
python scripts/repo_doctor.py fast
python scripts/launch_reproducible.py --seed 42 -- python scripts/toy_experiment_audit.py --run-dir outputs/toy_eegnet/onboarding
```

The toy run is a platform smoke test only. It proves the loader, job adapter, prediction writer, metric report, manifest, and audit path work; it is not a real EEG result.

