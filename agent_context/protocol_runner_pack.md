# Protocol Runner Pack

## Purpose
Work on P1/P2/P3 runner materialization, execution, and artifact-job wiring.

## Required Read Files
- `agent_protocols/AGENT_BRIEF.md`
- `docs/04_evaluation_protocols.md`
- `src/hust_bci_er/evaluation/protocols/runner.py`
- `src/hust_bci_er/evaluation/protocols/subject_splits.py`
- `scripts/run_evaluation_protocol.py`
- `scripts/run_route_job.py`
- `scripts/run_artifact_job.py`
- `tests/test_protocol_runner.py`

## Allowed Edit Scope
- `src/hust_bci_er/evaluation/protocols/`
- `scripts/run_evaluation_protocol.py`
- `scripts/run_route_job.py`
- `scripts/run_artifact_job.py`
- protocol runner tests

## Forbidden Edit Scope
- Route status
- Existing route results
- Split definition changes unless explicitly requested and reviewed
- Model internals unless validation points there

## Validation Commands
- `python -m pytest tests/test_protocol_runner.py -q`
- relevant CLI smoke/dry-run
- `python scripts/repo_doctor.py fast`

## Context Budget
Protocol runner files and targeted tests.

## Escalation Conditions
- Candidate-grade evidence semantics are involved.
- Evaluation protocol definitions would change.

## Issue Board Required
No unless the input is plan-driven or review-fix.
