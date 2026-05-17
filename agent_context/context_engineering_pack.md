# Context Engineering Pack

## Purpose
Maintain adaptive intake, context-pack selection, issue-board ingestion, session ledger, and agent-facing docs/tests.

## Required Read Files
- `agent_protocols/AGENT_BRIEF.md`
- `AGENTS.md`
- `agent_protocols/skill_router.md`
- `docs/目录怎么用.md`
- `tests/test_agent_protocols.py`

## Allowed Edit Scope
- `AGENTS.md`
- `agent_protocols/`
- `agent_context/`
- `scripts/agent_*.py`
- `tests/test_agent_intake_workflow.py`
- `tests/test_agent_protocols.py`
- short docs section in `docs/目录怎么用.md`

## Forbidden Edit Scope
- Model behavior
- Route status
- Experiment metrics
- Datasets
- Split definitions
- Candidate evidence
- Promotion evidence
- Existing route results

## Validation Commands
- `python -m pytest tests/test_agent_intake_workflow.py tests/test_agent_protocols.py -q`
- `python scripts/agent_intake.py --message "score-fusion：修 component map 的 whitening 绑定问题"`
- `python scripts/agent_intake.py --message "现在这个仓库后面该怎么做"`
- `python scripts/agent_context.py --task score-fusion`
- `python scripts/agent_context.py --task review-fix`
- `python scripts/repo_doctor.py fast`

## Context Budget
Brief, context-engineering pack, router docs, agent scripts, and tests.

## Escalation Conditions
- Request expands into route/evidence/model changes.
- Optional API-backed intake refinement changes safety flags instead of only refining mode/task family.

## Issue Board Required
No for direct context-engineering tasks unless the input is a long plan or review-fix request.
