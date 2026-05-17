# Adaptive Agent Intake Workflow

Agents must classify the user input before broad repository discovery. The
default is minimal context selected by task family.

## Modes

| Intake mode | Trigger signals | Allowed first actions | Issue board required | Code edits allowed | When to ask the human | Context budget |
|---|---|---|---|---|---|---|
| `plan-driven` | Large structured operation plan, AI review, S0/S1/S2 list, validation commands, multi-phase fix plan, many file paths | Save raw plan to a session directory, run `scripts/agent_plan_ingest.py`, create `issue_board.json/md`, then read selected context pack | Yes | After issue board creation | When issues conflict, require state changes, or evidence is missing | Brief + issue board + selected context pack + directly referenced files |
| `review-fix` | "fix PR review", unresolved review comments, requested changes, "修一下 PR #... review" | Fetch PR review/conversation comments if PR number is available, ingest user-provided review text if present, create issue board | Yes | After issue board creation | When comments conflict or require route/evidence/status changes | Brief + review-fix pack + issue board + referenced files |
| `direct-task` | Clear narrow task such as "score-fusion：修 whitening 绑定", "evidence-gate：修 promotion required rules", "model-backbone：给 FBSTCNet 加 guard" | Select task family, initialize session ledger with a short micro-plan, read selected context pack | Only if pack requires it | Yes | When the requested fix crosses forbidden scope or changes evidence semantics | Brief + selected context pack + directly referenced files |
| `planning-only` | "现在该怎么做", "分析一下路线", "给计划", "review only", "怎么看" | Read only bounded planning context and output a plan | No | No | If implementation is requested later | Brief + planning context only |
| `command-only` | "跑 fast gate", "dry-run 一下", a named check command | Run requested command and report output | No | No | If the command fails and fixes are requested | Brief + command output |
| `state-changing` | merge PR, promote route, change route status, delete evidence, modify split/evaluation protocol | Verify explicit audit-backed evidence; otherwise stop and plan only | Usually yes if continuing | Only with explicit evidence and human approval | If audit evidence is absent or ambiguous | Brief + evidence context only |
| `ambiguous` | Low confidence or mixed intent | Bounded triage only | No | No implementation edits | Ask a short clarification or propose next plan | Brief + at most one pack |

## Required Behavior

1. Run or emulate `scripts/agent_intake.py`.
2. Select the task family with `scripts/agent_context.py --task <task_family>`.
3. Read the selected context pack before implementation edits.
4. For `plan-driven` and `review-fix`, create an issue board before editing implementation files.
5. For `planning-only`, do not edit code.
6. For `command-only`, do not do broad discovery.
7. For `state-changing`, do not proceed without explicit audit evidence or human approval.

Deterministic intake is the default. Use `--deepseek` or
`AGENT_INTAKE_ENGINE=deepseek` only when external API refinement is explicitly
allowed; if the API is unavailable, `scripts/agent_intake.py` falls back to the
deterministic classifier unless `--require-deepseek` is set. Model output may
refine `mode`, `task_family`, `confidence`, and `reason`, but local code must
recompute issue-board flags, edit permission, ask-human flags, and selected
context pack. Because DeepSeek intake sends the user task text to an external
API, use deterministic intake for private review content unless the human
explicitly allows external API use.

## Session State

Use `scripts/agent_session.py` to persist:

- task family
- intake mode
- selected context pack
- issue board path
- files read
- commands run
- issue status
- validation status
- next action
