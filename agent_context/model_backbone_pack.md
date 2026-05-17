# Model Backbone Pack

## Purpose
Change model backbone/factory behavior with focused smoke coverage.

## Required Read Files
- `agent_protocols/AGENT_BRIEF.md`
- `src/hust_bci_er/models/factory.py`
- relevant file under `src/hust_bci_er/models/`
- `scripts/model_smoke.py`
- `tests/test_backbone_guards.py`

## Allowed Edit Scope
- `src/hust_bci_er/models/`
- `src/hust_bci_er/training/` only if adapter integration requires it
- model smoke/backbone tests

## Forbidden Edit Scope
- Route summaries
- Candidate or promotion evidence
- Dataset/split definitions
- Route status

## Validation Commands
- `python scripts/model_smoke.py`
- targeted model tests
- `python scripts/repo_doctor.py fast`

## Context Budget
Model factory, relevant backbone, and smoke tests. Avoid route summaries and evidence unless provenance tests fail.

## Escalation Conditions
- Audit/provenance failure points outside model code.
- Any route status or evidence update is requested.

## Issue Board Required
No unless the input is plan-driven or review-fix.
