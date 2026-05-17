from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from scripts.agent_context import CONTEXT_PACKS, context_for_task
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from agent_context import CONTEXT_PACKS, context_for_task


MODE_ORDER = [
    "plan-driven",
    "review-fix",
    "direct-task",
    "planning-only",
    "command-only",
    "state-changing",
    "ambiguous",
]


TASK_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("score-fusion", ("score-fusion", "score_fusion", "score fusion", "component map", "component-map", "whitening", "融合")),
    ("evidence-gate", ("evidence-gate", "evidence gate", "audit", "manifest", "promotion required", "证据", "审计")),
    ("route-config", ("route-config", "route config", "route 配置", "configs/routes", "validate_route")),
    ("model-backbone", ("model-backbone", "backbone", "model factory", "factory", "fbstcnet", "conformer", "deformer", "模型")),
    ("summary-report", ("summary-report", "summary report", "route summary", "summarize", "报告", "summary")),
    ("protocol-runner", ("protocol-runner", "protocol runner", "run_evaluation_protocol", "p1", "p2", "p3", "materialize", "协议")),
    ("review-fix", ("review fix", "unresolved review", "conversation review", "pr review", "修 review")),
    ("context-engineering", ("context-engineering", "context engineering", "agent intake", "context pack", "issue board", "minimal context")),
    ("repo-fast", ("repo fast", "fast gate", "repo_doctor.py fast", "仓库健康")),
]


PLAN_SIGNALS = (
    "s0",
    "s1",
    "s2",
    "validation",
    "issue",
    "blocking",
    "no-go",
    "需要补",
    "仍然阻断",
    "阻断",
    "phase",
    "任务",
)
REVIEW_WORDS = ("review", "conversation", "unresolved", "comment", "comments", "pr", "修一下", "修复")
PLANNING_WORDS = ("what should we do", "怎么做", "该怎么做", "分析", "计划", "给计划", "review only", "怎么看")
EDIT_WORDS = ("fix", "修", "实现", "改", "edit", "implement", "处理")
COMMAND_ONLY_PATTERNS = (
    r"^\s*(跑|run|执行)\s+.*(fast gate|repo fast|repo_doctor\.py fast|dry-run|dry run|smoke|pytest|check)",
    r"^\s*python\s+scripts/[^\n]+$",
)
STATE_CHANGING_WORDS = (
    "merge pr",
    "合并",
    "promote",
    "promoted",
    "route status",
    "改成 promoted",
    "change route status",
    "delete evidence",
    "删除 evidence",
    "删除证据",
    "modify split",
    "change split",
    "修改 split",
    "alter splits",
    "evaluation protocol",
    "评估协议",
)
AUDIT_EVIDENCE_WORDS = ("audit report", "audit pass", "candidate gate pass", "promoted gate pass", "审计通过", "audit evidence")


def read_message(args: argparse.Namespace) -> str:
    if args.message_file:
        return Path(args.message_file).read_text(encoding="utf-8")
    return args.message or ""


def count_paths(text: str) -> int:
    pattern = r"(?:^|\s)(?:[\w.-]+/[\w./-]+|[\w.-]+\\[\w.\\/-]+|\w+\.py|\w+\.md|\w+\.yaml|\w+\.json)"
    return len(re.findall(pattern, text))


def count_commands(text: str) -> int:
    return len(re.findall(r"(?m)^\s*(?:python|pytest|git|gh)\s+[^\n]+", text))


def detect_task_family(text: str) -> str:
    lowered = text.lower()
    for family, keywords in TASK_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return family
    return "review" if "review" in lowered else "repo-fast" if "fast" in lowered else "context-engineering" if "agent" in lowered else "repo-fast"


def has_pr_review_signal(text: str) -> bool:
    lowered = text.lower()
    has_pr = re.search(r"\bpr\s*#?\d+\b", lowered) is not None or re.search(r"\bpull request\s*#?\d+\b", lowered) is not None
    has_review = any(word in lowered for word in REVIEW_WORDS)
    return bool(has_pr and has_review)


def has_plan_signal(text: str) -> bool:
    lowered = text.lower()
    signal_count = sum(1 for signal in PLAN_SIGNALS if signal in lowered)
    return signal_count >= 2 or count_paths(text) >= 4 or count_commands(text) >= 2 or len(text.splitlines()) >= 18


def has_state_change_signal(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in STATE_CHANGING_WORDS)


def has_audit_evidence(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in AUDIT_EVIDENCE_WORDS)


def is_command_only(text: str) -> bool:
    stripped = text.strip()
    if len(stripped.splitlines()) > 3:
        return False
    lowered = stripped.lower()
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in COMMAND_ONLY_PATTERNS):
        return True
    return lowered in {"跑 fast gate", "run fast gate", "跑 repo fast gate", "dry-run 一下", "dry run"}


def asks_planning_only(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in PLANNING_WORDS) and not any(word in lowered for word in EDIT_WORDS)


def known_direct_task(text: str) -> bool:
    lowered = text.lower()
    return any(any(keyword in lowered for keyword in keywords) for family, keywords in TASK_KEYWORDS if family not in {"repo-fast", "review-fix"})


def classify(text: str) -> dict[str, Any]:
    text = text or ""
    lowered = text.lower()
    mode = "ambiguous"
    confidence = 0.35
    reason = "No strong deterministic trigger matched."

    if has_state_change_signal(text) and not has_audit_evidence(text):
        mode = "state-changing"
        confidence = 0.9
        reason = "State-changing request lacks explicit audit evidence."
    elif is_command_only(text):
        mode = "command-only"
        confidence = 0.88
        reason = "Input asks only to run a named check/command."
    elif has_pr_review_signal(text):
        mode = "review-fix"
        confidence = 0.92
        reason = "Input mentions PR plus review/conversation/unresolved fix signals."
    elif asks_planning_only(text):
        mode = "planning-only"
        confidence = 0.86
        reason = "Input asks for analysis or plan and does not request edits."
    elif has_plan_signal(text):
        mode = "plan-driven"
        confidence = 0.9
        reason = "Input contains structured plan/review signals, file hints, or multiple commands."
    elif known_direct_task(text):
        mode = "direct-task"
        confidence = 0.78
        reason = "Input has a known narrow task-family keyword."

    task_family = detect_task_family(text)
    if mode == "command-only":
        task_family = "repo-fast"
    if mode == "review-fix":
        task_family = "review-fix"
    if mode == "planning-only" and task_family == "repo-fast":
        task_family = "review"
    if mode == "state-changing" and "promot" in lowered:
        task_family = "evidence-gate"

    context = context_for_task(task_family)
    needs_issue_board = mode in {"plan-driven", "review-fix"} or task_family == "review-fix"
    needs_plan_ingest = mode == "plan-driven"
    needs_pr_review_fetch = mode == "review-fix"
    should_edit = mode in {"plan-driven", "review-fix", "direct-task"} and not (mode == "state-changing")
    should_ask_human = mode in {"ambiguous", "state-changing"}
    if mode == "planning-only" or mode == "command-only":
        should_edit = False
    if mode == "state-changing" and has_audit_evidence(text):
        should_edit = False
        should_ask_human = False

    initial_action_by_mode = {
        "plan-driven": "Save raw plan to scratch/agent_sessions, run scripts/agent_plan_ingest.py, create issue_board, then read the selected context pack.",
        "review-fix": "Fetch PR review threads with scripts/agent_review_inbox.py or ingest provided review text, create issue_board, then read the selected context pack.",
        "direct-task": "Select task family, initialize session ledger with a micro-plan, then read the selected context pack.",
        "planning-only": "Do bounded reading from the selected context pack and return a plan without code edits.",
        "command-only": "Run the requested command directly and report the result.",
        "state-changing": "Stop for audit-backed evidence or produce a plan only before changing repository state.",
        "ambiguous": "Do bounded triage and ask a short clarification or recommend the next plan.",
    }

    return {
        "mode": mode,
        "task_family": task_family,
        "confidence": confidence,
        "context_pack": context["context_pack_path"],
        "needs_issue_board": needs_issue_board,
        "needs_plan_ingest": needs_plan_ingest,
        "needs_pr_review_fetch": needs_pr_review_fetch,
        "should_edit": should_edit,
        "should_ask_human": should_ask_human,
        "initial_action": initial_action_by_mode[mode],
        "reason": reason,
    }


def _json_from_text(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("DeepSeek response JSON must be an object")
    return value


def _recompute_flags(result: dict[str, Any], original_text: str) -> dict[str, Any]:
    mode = str(result.get("mode") or "ambiguous")
    task_family = str(result.get("task_family") or detect_task_family(original_text))
    if mode not in MODE_ORDER:
        mode = "ambiguous"
    if task_family not in CONTEXT_PACKS:
        task_family = detect_task_family(original_text)
        if task_family not in CONTEXT_PACKS:
            task_family = "repo-fast"

    context = context_for_task(task_family)
    reason = str(result.get("reason") or "DeepSeek refined the deterministic intake result.")
    try:
        confidence = float(result.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    needs_issue_board = mode in {"plan-driven", "review-fix"} or task_family == "review-fix"
    needs_plan_ingest = mode == "plan-driven"
    needs_pr_review_fetch = mode == "review-fix"
    should_edit = mode in {"plan-driven", "review-fix", "direct-task"}
    should_ask_human = mode in {"ambiguous", "state-changing"}
    if mode in {"planning-only", "command-only", "state-changing"}:
        should_edit = False

    initial_action_by_mode = {
        "plan-driven": "Save raw plan to scratch/agent_sessions, run scripts/agent_plan_ingest.py, create issue_board, then read the selected context pack.",
        "review-fix": "Fetch PR review threads with scripts/agent_review_inbox.py or ingest provided review text, create issue_board, then read the selected context pack.",
        "direct-task": "Select task family, initialize session ledger with a micro-plan, then read the selected context pack.",
        "planning-only": "Do bounded reading from the selected context pack and return a plan without code edits.",
        "command-only": "Run the requested command directly and report the result.",
        "state-changing": "Stop for audit-backed evidence or produce a plan only before changing repository state.",
        "ambiguous": "Do bounded triage and ask a short clarification or recommend the next plan.",
    }
    return {
        "mode": mode,
        "task_family": task_family,
        "confidence": confidence,
        "context_pack": context["context_pack_path"],
        "needs_issue_board": needs_issue_board,
        "needs_plan_ingest": needs_plan_ingest,
        "needs_pr_review_fetch": needs_pr_review_fetch,
        "should_edit": should_edit,
        "should_ask_human": should_ask_human,
        "initial_action": initial_action_by_mode[mode],
        "reason": reason,
    }


def classify_with_deepseek(
    text: str,
    deterministic_result: dict[str, Any],
    *,
    api_key: str,
    model: str,
    base_url: str,
    timeout: float,
) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    system = (
        "You refine a deterministic repository intake classification. "
        "Return JSON only. Allowed modes: "
        + ", ".join(MODE_ORDER)
        + ". Allowed task_family values: "
        + ", ".join(sorted(CONTEXT_PACKS))
        + ". Keep state-changing requests non-editable unless explicit audit evidence is present. "
        "Prefer deterministic_result when unsure."
    )
    user = {
        "user_message": text,
        "deterministic_result": deterministic_result,
        "required_keys": [
            "mode",
            "task_family",
            "confidence",
            "reason",
        ],
    }
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"DeepSeek API request failed: {exc}") from exc

    choices = response_payload.get("choices") if isinstance(response_payload, dict) else None
    if not choices:
        raise RuntimeError("DeepSeek API response did not include choices")
    content = choices[0].get("message", {}).get("content", "")
    refined = _json_from_text(str(content))
    merged = _recompute_flags(refined, text)
    merged["deepseek"] = {
        "used": True,
        "model": model,
        "base_url": base_url,
    }
    return merged


def print_human(result: dict[str, Any]) -> None:
    print(f"mode: {result['mode']}")
    print(f"task_family: {result['task_family']}")
    print(f"context_pack: {result['context_pack']}")
    print(f"confidence: {result['confidence']}")
    print(f"should_edit: {str(result['should_edit']).lower()}")
    if result.get("deepseek", {}).get("used"):
        print(f"deepseek_model: {result['deepseek']['model']}")
    elif result.get("deepseek", {}).get("fallback"):
        print(f"deepseek_fallback: {result['deepseek']['fallback']}")
    print(f"initial_action: {result['initial_action']}")
    print(f"reason: {result['reason']}")


def maybe_refine_with_deepseek(
    message: str,
    deterministic_result: dict[str, Any],
    *,
    engine: str,
    api_key: str | None,
    model: str,
    base_url: str,
    timeout: float,
    require_deepseek: bool,
) -> dict[str, Any]:
    if engine == "deterministic":
        return deterministic_result
    if engine == "auto" and not api_key:
        return deterministic_result
    if not api_key:
        if require_deepseek:
            raise RuntimeError("DeepSeek intake requires DEEPSEEK_API_KEY in the environment")
        result = dict(deterministic_result)
        result["deepseek"] = {
            "used": False,
            "model": model,
            "base_url": base_url,
            "fallback": "missing DEEPSEEK_API_KEY; used deterministic intake",
        }
        return result
    try:
        return classify_with_deepseek(
            message,
            deterministic_result,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )
    except RuntimeError as exc:
        if require_deepseek:
            raise
        result = dict(deterministic_result)
        result["deepseek"] = {
            "used": False,
            "model": model,
            "base_url": base_url,
            "fallback": f"{exc}; used deterministic intake",
        }
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify user input into the repository adaptive agent intake workflow.")
    parser.add_argument("--message", help="User message to classify.")
    parser.add_argument("--message-file", type=Path, help="File containing user message to classify.")
    parser.add_argument("--json", action="store_true", help="Also print JSON payload.")
    parser.add_argument("--list-modes", action="store_true", help="List supported intake modes.")
    parser.add_argument(
        "--intake-engine",
        choices=("auto", "deepseek", "deterministic"),
        default=os.environ.get("AGENT_INTAKE_ENGINE", "auto"),
        help="auto prefers DeepSeek when DEEPSEEK_API_KEY is set and falls back to deterministic intake.",
    )
    parser.add_argument("--deepseek", action="store_true", help="Prefer DeepSeek intake; falls back unless --require-deepseek is set.")
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic local intake.")
    parser.add_argument("--require-deepseek", action="store_true", help="Fail instead of falling back when DeepSeek intake is unavailable.")
    parser.add_argument("--deepseek-model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--deepseek-base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--deepseek-timeout", type=float, default=20.0)
    args = parser.parse_args(argv)

    if args.list_modes:
        for mode in MODE_ORDER:
            print(mode)
        return 0
    if not args.message and not args.message_file:
        parser.error("--message or --message-file is required unless --list-modes is used")
    message = read_message(args)
    result = classify(message)
    engine = args.intake_engine
    if args.deepseek:
        engine = "deepseek"
    if args.deterministic:
        engine = "deterministic"
    if engine not in {"auto", "deepseek", "deterministic"}:
        parser.error("AGENT_INTAKE_ENGINE must be one of: auto, deepseek, deterministic")
    try:
        result = maybe_refine_with_deepseek(
            message,
            result,
            engine=engine,
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            model=args.deepseek_model,
            base_url=args.deepseek_base_url,
            timeout=args.deepseek_timeout,
            require_deepseek=args.require_deepseek,
        )
    except RuntimeError as exc:
        parser.error(str(exc))
    print_human(result)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
