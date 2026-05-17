# Review Fix Pack

## Purpose
Fix actionable PR review or conversation feedback with minimal context.

## Required Read Files
- `agent_protocols/AGENT_BRIEF.md`
- this pack
- `scripts/agent_review_inbox.py`
- `scripts/agent_plan_ingest.py`
- `scripts/agent_session.py`
- generated `issue_board.json` and `issue_board.md`

## Allowed Edit Scope
- Files referenced by the issue board
- Tests required to prove those fixes

## Forbidden Edit Scope
- Unrelated files
- Route status, candidate evidence, or promotion evidence unless the issue board explicitly requires it and audit evidence supports it
- Resolving PR threads before fixes are pushed

## Validation Commands
- Targeted tests for fixed review issues
- `python scripts/repo_doctor.py fast`

## Context Budget
Brief, review-fix pack, generated issue board, and selected files only.

## Escalation Conditions
- Review comments conflict.
- A fix requires state-changing route/evidence/protocol updates.
- GitHub review fetch fails and no review-file fallback is available.

## Issue Board Required
Yes, before implementation edits.
