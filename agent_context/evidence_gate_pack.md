# Evidence Gate Pack

## Purpose
Adjust audit, schema, manifest, and gate behavior without touching model internals.

## Required Read Files
- `agent_protocols/AGENT_BRIEF.md`
- `agent_protocols/experiment_audit.md`
- `scripts/audit_experiment.py`
- `scripts/repo_doctor.py`
- relevant files under `src/hust_bci_er/audit/`
- `tests/test_audit_gate_behavior.py`

## Allowed Edit Scope
- `scripts/audit_experiment.py`
- `scripts/repo_doctor.py`
- `src/hust_bci_er/audit/`
- audit/gate tests

## Forbidden Edit Scope
- Model backbones
- Existing route results
- Candidate/promotion evidence content
- Split or evaluation protocol semantics unless explicitly approved

## Validation Commands
- `python -m pytest tests/test_audit_gate_behavior.py -q`
- `python scripts/repo_doctor.py fast`

## Context Budget
Audit/schema/manifest code and directly failing tests only. Avoid model backbones.

## Escalation Conditions
- A fix requires changing metric semantics.
- A fix would change route status or evidence.

## Issue Board Required
No unless the input is plan-driven or review-fix.
