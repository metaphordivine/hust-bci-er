# Review Pack

## Purpose
Perform read-only PR/code review in Simplified Chinese.

## Required Read Files
- `agent_protocols/AGENT_BRIEF.md`
- diff or files explicitly provided by the review request

## Allowed Edit Scope
- None.

## Forbidden Edit Scope
- Implementation edits
- Resolving PR review threads
- Pushing commits

## Validation Commands
- None unless review explicitly asks to run checks.

## Context Budget
Review pack, diff, and directly referenced files.

## Escalation Conditions
- The user asks to fix the review.
- The review identifies state-changing implications.

## Issue Board Required
No for read-only review.
