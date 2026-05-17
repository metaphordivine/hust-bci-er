# Agent Brief

This is the short universal brief for repository agents.

## Default

Minimal context is the repository default. Humans do not need to say "use
minimal context".

Before reading large protocol documents, agents must classify the user input.
Use `scripts/agent_intake.py`, then select a context pack with
`scripts/agent_context.py --task <task_family>`.

`scripts/agent_intake.py` uses DeepSeek-first intake when `DEEPSEEK_API_KEY` is
available, with deterministic fallback when the key or API is unavailable. Local
code must still recompute safety flags, context pack selection, and red-line
behavior after the model response. Use `--deterministic` for offline or CI-only
local classification.

DeepSeek-first intake may transmit user task text to the external DeepSeek API.
Do not use it for private review content, confidential PR comments, or sensitive
plans unless the human explicitly allows external API use; use `--deterministic`
in those cases.

Read full protocol documents only when the selected context pack requires them,
validation points there, or the task explicitly edits that protocol.

## Intake Rules

Classify the request into one intake mode:

- `plan-driven`
- `review-fix`
- `direct-task`
- `planning-only`
- `command-only`
- `state-changing`
- `ambiguous`

For long plans, AI reviews, S0/S1/S2 lists, or multi-phase fix plans, create an
issue board before implementation edits.

For PR review-fix work, fetch or ingest review comments into an issue board
before implementation edits. Do not resolve review threads until the underlying
fix is pushed.

For direct narrow tasks, create a short session micro-plan and read only the
selected context pack plus directly referenced files.

For planning-only and command-only tasks, do not perform broad discovery.

## Absolute Red Lines

- Do not use public/private labels or leaderboard feedback.
- Do not silently change split definitions or evaluation protocols.
- Do not use `subject_id`, `trial_id`, `pseudo_trial_id`, or filenames as model features.
- Do not commit checkpoints, raw outputs, full prediction tables, or temporary dumps.
- Do not change route status unless an audit report supports it.

## Completion Rules

- Run targeted tests first.
- Run `python scripts/repo_doctor.py fast` before final handoff unless the task is read-only or explicitly planning-only.
- If validation fails, report the failure and do not upgrade route status.
- Keep generated session state under `scratch/agent_sessions/` or another ignored path.
- Final reports for this repository should be in Simplified Chinese.

## Safe Defaults

- Prefer existing scripts and foundation modules over one-off scripts.
- Keep edits inside the selected context pack's allowed scope.
- Escalate when a task touches route status, candidate evidence, promotion evidence, split definitions, or evaluation protocol semantics.
