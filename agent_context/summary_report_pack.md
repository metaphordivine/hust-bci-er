# Summary Report Pack

## Purpose
Maintain protocol summaries, route summaries, and report consistency checks.

## Required Read Files
- `agent_protocols/AGENT_BRIEF.md`
- `src/hust_bci_er/evaluation/protocols/summary.py`
- `scripts/summarize_protocol_run.py`
- `scripts/check_summary_consistency.py`
- `tests/test_summary_consistency.py`
- `tests/test_protocol_runner.py`

## Allowed Edit Scope
- summary/report scripts and modules
- summary consistency tests

## Forbidden Edit Scope
- Metric values
- Route status
- Candidate or promotion evidence content
- Existing route results

## Validation Commands
- `python -m pytest tests/test_summary_consistency.py tests/test_protocol_runner.py -q`
- `python scripts/repo_doctor.py fast`

## Context Budget
Summary/report pack and directly related tests.

## Escalation Conditions
- Summary would imply a route status upgrade.
- Manifest/audit schema needs to change.

## Issue Board Required
No unless the input is plan-driven or review-fix.
