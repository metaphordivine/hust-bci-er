# Score Fusion Pack

## Purpose
Work on score-fusion routes, component score export, component maps, and fusion artifact evidence.

## Required Read Files
- `agent_protocols/AGENT_BRIEF.md`
- `src/hust_bci_er/config/component_map.py`
- `src/hust_bci_er/inference/clean_score_routes.py`
- `src/hust_bci_er/inference/score_route_assembly.py`
- `scripts/run_score_fusion_routes.py`
- `scripts/export_component_scores.py`
- `tests/test_score_fusion_runner.py`
- `tests/test_score_route_assembly.py`

## Allowed Edit Scope
- `src/hust_bci_er/config/component_map.py`
- `src/hust_bci_er/inference/`
- `scripts/run_score_fusion_routes.py`
- `scripts/export_component_scores.py`
- score-fusion tests

## Forbidden Edit Scope
- Unrelated model backbone internals
- Dataset manifests and split definitions
- Route status
- Existing route results or candidate evidence

## Validation Commands
- `python -m pytest tests/test_score_fusion_runner.py tests/test_score_route_assembly.py -q`
- score-fusion dry-run or focused assembly command
- `python scripts/repo_doctor.py fast`

## Context Budget
Score-fusion runner, component map, score assembly code, and focused tests. Avoid unrelated model internals unless validation failure points there.

## Escalation Conditions
- Component evidence is missing, synthetic, or untrusted.
- A fix would require retraining component models or changing dataset/split evidence.

## Issue Board Required
No unless the input is plan-driven or review-fix.
