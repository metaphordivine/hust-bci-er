# Repo Fast Pack

## Purpose
Run or inspect the repository fast gate with minimal context.

## Required Read Files
- `agent_protocols/AGENT_BRIEF.md`
- this pack

## Allowed Edit Scope
- None by default.

## Forbidden Edit Scope
- Implementation files unless the user asks for a fix after a failed check.
- Route status, evidence, datasets, and split definitions.

## Validation Commands
- `python scripts/repo_doctor.py fast`

## Context Budget
Brief, this pack, and command output only.

## Escalation Conditions
- Fast gate fails and the user asks for fixes.
- The request becomes state-changing.

## Issue Board Required
No.
