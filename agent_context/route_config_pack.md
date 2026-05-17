# Route Config Pack

## Purpose
Add or adjust route configs without touching results or evidence.

## Required Read Files
- `agent_protocols/AGENT_BRIEF.md`
- `configs/statuses.yaml`
- `src/hust_bci_er/config/schema.py`
- directly referenced route config

## Allowed Edit Scope
- `configs/routes/models/`
- route schema tests when validation requires them

## Forbidden Edit Scope
- Dataset manifests
- Split definitions
- Route status changes without supporting audit report
- Existing route results or evidence

## Validation Commands
- `python scripts/validate_route.py --all`
- `python scripts/repo_doctor.py fast`

## Context Budget
Route config pack, the changed route config, and schema/test references.

## Escalation Conditions
- Request changes evaluation protocol or split definitions.
- Request promotes or demotes route lifecycle status.

## Issue Board Required
No unless the input is plan-driven or review-fix.
