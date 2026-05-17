from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from scripts.agent_env import configured_env
    from scripts.agent_plan_ingest import COMMAND_RE, PATH_RE, dedupe_issues, extract_expected_fix, parse_text, write_issue_board
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from agent_env import configured_env
    from agent_plan_ingest import COMMAND_RE, PATH_RE, dedupe_issues, extract_expected_fix, parse_text, write_issue_board


def fetch_pr_review_threads(pr_number: int) -> dict[str, Any]:
    if shutil.which("gh") is None:
        raise RuntimeError("GitHub CLI `gh` is not available")
    query = """
    query($owner:String!, $name:String!, $number:Int!) {
      repository(owner:$owner, name:$name) {
        pullRequest(number:$number) {
          reviewThreads(first:100) {
            nodes {
              id
              isResolved
              isOutdated
              comments(first:20) {
                nodes {
                  body
                  path
                  line
                  author { login }
                  createdAt
                }
              }
            }
          }
          comments(first:100) {
            nodes {
              id
              url
              body
              author { login }
              createdAt
            }
          }
        }
      }
    }
    """
    repo = subprocess.run(
        ["gh", "repo", "view", "--json", "nameWithOwner"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    )
    repo_data = json.loads(repo.stdout)
    name_with_owner = str(repo_data["nameWithOwner"])
    owner, name = name_with_owner.split("/", 1)
    proc = subprocess.run(
        [
            "gh",
            "api",
            "graphql",
            "-f",
            f"owner={owner}",
            "-f",
            f"name={name}",
            "-F",
            f"number={int(pr_number)}",
            "-f",
            f"query={query}",
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout)


def review_text_from_payload(payload: dict[str, Any]) -> str:
    pr = payload.get("data", {}).get("repository", {}).get("pullRequest", {})
    chunks: list[str] = []
    for thread in pr.get("reviewThreads", {}).get("nodes", []):
        if not isinstance(thread, dict):
            continue
        if thread.get("isResolved"):
            continue
        for comment in thread.get("comments", {}).get("nodes", []):
            if not isinstance(comment, dict):
                continue
            path = comment.get("path") or ""
            line = comment.get("line") or ""
            body = comment.get("body") or ""
            chunks.append(f"## Review thread {thread.get('id')} {path}:{line}\n\n{body}\n")
    for comment in pr.get("comments", {}).get("nodes", []):
        if isinstance(comment, dict) and comment.get("body"):
            chunks.append(f"## PR comment\n\n{comment['body']}\n")
    return "\n".join(chunks)


def _short_thread_id(thread_id: str) -> str:
    safe = "".join(ch for ch in thread_id if ch.isalnum())
    if safe:
        return safe[-10:]
    return hashlib.sha1(thread_id.encode("utf-8")).hexdigest()[:10]


def _stable_short_id(value: str) -> str:
    safe = "".join(ch for ch in value if ch.isalnum())
    if safe:
        return safe[-10:]
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def strip_useful_suffix(body: str) -> str:
    return re.sub(r"(?is)\n?\s*Useful\? React with.*$", "", body).strip()


def severity_from_badge(body: str) -> str | None:
    lowered = body.lower()
    if re.search(r"(?:p0|p1)\s+badge|badge/(?:p0|p1)(?:\b|-)|p1-orange", lowered):
        return "S0"
    if re.search(r"p2\s+badge|badge/p2(?:\b|-)|p2-yellow", lowered):
        return "S1"
    if re.search(r"p3\s+badge|badge/p3(?:\b|-)", lowered):
        return "S2"
    return None


def infer_review_severity(body: str) -> str:
    badge_severity = severity_from_badge(body)
    if badge_severity is not None:
        return badge_severity
    lowered = body.lower()
    if any(marker in lowered for marker in ("🔴", "[p0]", "p0:", "[p1]", "p1:", "s0", "高风险", "blocking", "no-go", "阻断", "阻塞")):
        return "S0"
    if any(marker in lowered for marker in ("🟠", "🟡", "[p2]", "p2:", "s1", "中风险", "中级风险")):
        return "S1"
    if any(marker in lowered for marker in ("🔵", "ℹ️", "[p3]", "p3:", "s2", "低风险", "信息性")):
        return "S2"
    return "S1"


def extract_review_title(body: str) -> str:
    for line in body.splitlines():
        clean = line.strip(" #*-`\t")
        if not clean:
            continue
        clean = clean.removeprefix("[P0]").removeprefix("[P1]").removeprefix("[P2]").removeprefix("[P3]").strip(" ：:-")
        return clean[:140] or "Review thread"
    return "Review thread"


ACTIONABLE_PR_COMMENT_WORDS = (
    "fix",
    "修",
    "处理",
    "解决",
    "address",
    "resolve",
    "unresolved",
    "requested changes",
    "需要",
    "请",
    "must",
    "should",
    "blocking",
    "no-go",
    "阻断",
)


def is_actionable_pr_comment(body: str) -> bool:
    lowered = body.lower()
    return any(word in lowered for word in ACTIONABLE_PR_COMMENT_WORDS)


def _author_login(comment: dict[str, Any]) -> str:
    author = comment.get("author")
    if isinstance(author, dict):
        return str(author.get("login") or "")
    return ""


def strip_copilot_quote(body: str) -> str:
    lines = body.splitlines()
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped:
            idx += 1
            continue
        if stripped.startswith(">"):
            idx += 1
            continue
        break
    return "\n".join(lines[idx:]).strip() or body.strip()


def is_copilot_trigger_comment(body: str) -> bool:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not lines or len(lines) > 2:
        return False
    return all(line.lstrip("> ").lower().startswith("@copilot") for line in lines)


def classify_pr_comment(comment: dict[str, Any]) -> str:
    body = str(comment.get("body") or "").strip()
    lowered = body.lower()
    author = _author_login(comment).lower()
    unquoted = strip_copilot_quote(body)
    if is_copilot_trigger_comment(body):
        return "trigger_comment"
    if body.lstrip().lower().startswith(("已处理", "已修复", "验证", "已验证", "fixed", "done")):
        return "human_fix_report"
    if "copilot" in author or (lowered.startswith("> @copilot") and unquoted != body.strip()):
        return "copilot_digest_comment"
    if is_actionable_pr_comment(body):
        return "plain_actionable_comment"
    return "plain_comment"


def has_no_open_issue_conclusion(body: str) -> bool:
    return any(phrase in body for phrase in ("未发现阻塞", "可合入", "当前没有新的阻塞问题", "未发现新的阻塞问题"))


def _extract_files(text: str) -> list[str]:
    files: set[str] = set()
    for match in re.findall(r"`([^`]+\.(?:py|md|yaml|yml|json|toml|txt))`", text):
        files.add(match.strip())
    for match in PATH_RE.findall(text):
        files.add(match.rstrip(".,;:)"))
    return sorted(files)


def _make_digest_issue(
    *,
    pr: int,
    comment_ref: str,
    index: int,
    source: str,
    title: str,
    problem: str,
    severity_text: str = "",
    files_hint: list[str] | None = None,
) -> dict[str, Any]:
    issue_text = f"{severity_text}\n{title}\n{problem}".strip()
    return {
        "id": f"PR{pr}-COPILOT-{_stable_short_id(comment_ref)}-{index}",
        "source": source,
        "kind": "copilot_digest_comment",
        "severity": infer_review_severity(issue_text),
        "title": title.strip(" ：:-")[:140] or "Copilot review issue",
        "problem": strip_useful_suffix(problem),
        "expected_fix": extract_expected_fix(problem),
        "files_hint": files_hint if files_hint is not None else _extract_files(issue_text),
        "validation_hint": [cmd.strip() for cmd in COMMAND_RE.findall(problem)],
        "status": "open",
    }


def _parse_digest_table(body: str, *, pr: int, source: str, comment_ref: str, start_index: int) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    lines = body.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not (line.startswith("|") and "优先级" in line and "位置" in line and "问题" in line):
            idx += 1
            continue
        headers = [part.strip() for part in line.strip("|").split("|")]
        idx += 1
        if idx < len(lines) and re.fullmatch(r"\s*\|?[\s:|\-]+\|?\s*", lines[idx]):
            idx += 1
        while idx < len(lines) and lines[idx].strip().startswith("|"):
            cells = [part.strip() for part in lines[idx].strip().strip("|").split("|")]
            row = dict(zip(headers, cells, strict=False))
            priority = row.get("优先级", "")
            location = row.get("位置", "")
            problem = row.get("问题", "") or "Copilot review issue"
            issues.append(
                _make_digest_issue(
                    pr=pr,
                    comment_ref=comment_ref,
                    index=start_index + len(issues),
                    source=source,
                    title=problem,
                    problem=lines[idx].strip(),
                    severity_text=priority,
                    files_hint=_extract_files(location),
                )
            )
            idx += 1
        continue
    return issues


def _line_heading_issue(line: str) -> tuple[str, str, str, list[str]] | None:
    clean = line.strip()
    clean = clean[1:].strip() if clean.startswith(">") else clean
    heading = re.match(r"^#{1,6}\s*(?P<risk>(?:🔴|🟠|🟡|🔵|ℹ️)?\s*(?:P[0-3]|S[0-2]|高风险|中风险|中级风险|低风险|信息性|Blocking|No-Go|阻塞|阻断))\s*[:：/\-]?\s*(?P<title>.*)$", clean, flags=re.IGNORECASE)
    if heading:
        title = heading.group("title").strip() or heading.group("risk").strip()
        return heading.group("risk"), title, clean, _extract_files(clean)
    bold_risk = re.match(r"^\*\*\s*(?P<risk>(?:⚠️|🔴|🟠|🟡|🔵|ℹ️)?\s*(?:P[0-3]|S[0-2]|高风险|中风险|中级风险|低风险|信息性|Blocking|No-Go|阻塞|阻断))\s*[：:-]\s*(?P<title>.+?)\s*\*\*$", clean, flags=re.IGNORECASE)
    if bold_risk:
        return bold_risk.group("risk"), bold_risk.group("title"), clean, _extract_files(clean)
    numbered = re.match(r"^\*\*\s*(?:\d+[.)]\s*)?(?:`(?P<path>[^`]+)`\s*[：:-]\s*)?(?P<title>[^*]+?)\s*\*\*$", clean)
    if numbered and (numbered.group("path") or re.match(r"^\*\*\s*\d+[.)]", clean)):
        path = numbered.group("path")
        files = [path] if path else _extract_files(clean)
        return "", numbered.group("title"), clean, files
    return None


def _is_risk_section_heading(severity_text: str, title: str) -> bool:
    clean_risk = severity_text.strip(" ：:-")
    clean_title = title.strip(" ：:-")
    return bool(clean_risk) and (not clean_title or clean_title == clean_risk)


def parse_copilot_digest_comment(body: str, pr: int, comment_ref: str) -> list[dict[str, Any]]:
    cleaned = strip_useful_suffix(strip_copilot_quote(body))
    source = f"pr:{pr}:copilot_digest:{comment_ref}"
    table_issues = _parse_digest_table(cleaned, pr=pr, source=source, comment_ref=comment_ref, start_index=1)
    if table_issues:
        return table_issues
    if has_no_open_issue_conclusion(cleaned) and not any(_line_heading_issue(line) for line in cleaned.splitlines()):
        return []

    heading_matches: list[tuple[int, str, str, str, list[str]]] = []
    current_risk = ""
    for idx, line in enumerate(cleaned.splitlines()):
        parsed = _line_heading_issue(line)
        if parsed is None:
            continue
        severity_text, title, raw, files = parsed
        if _is_risk_section_heading(severity_text, title):
            current_risk = severity_text
            continue
        if not severity_text and current_risk:
            severity_text = current_risk
        heading_matches.append((idx, severity_text, title, raw, files))
    if not heading_matches:
        if has_no_open_issue_conclusion(cleaned):
            return []
        return []

    lines = cleaned.splitlines()
    issues: list[dict[str, Any]] = []
    for issue_idx, (line_idx, severity_text, title, raw, files) in enumerate(heading_matches, start=1):
        end = heading_matches[issue_idx][0] if issue_idx < len(heading_matches) else len(lines)
        problem = "\n".join([raw, *lines[line_idx + 1 : end]]).strip()
        issues.append(
            _make_digest_issue(
                pr=pr,
                comment_ref=comment_ref,
                index=issue_idx,
                source=source,
                title=title,
                problem=problem,
                severity_text=severity_text,
                files_hint=files or None,
            )
        )
    return issues


def looks_like_copilot_digest(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    if not stripped:
        return False
    if stripped.startswith("> @copilot"):
        return True
    if "copilot" in lowered and any(marker in stripped for marker in ("###", "| 优先级 |", "**1.", "⚠️", "🔴", "🟠", "🟡")):
        return True
    digest_markers = sum(1 for marker in ("### 🔴", "### 🟠", "### 🟡", "| 优先级 |", "**1.", "> **⚠️") if marker in stripped)
    return digest_markers >= 2


def _json_from_text(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("DeepSeek digest response JSON must be an object")
    return value


def parse_copilot_digest_comment_with_deepseek(
    body: str,
    pr: int,
    comment_ref: str,
    *,
    api_key: str,
    model: str,
    base_url: str,
    timeout: float,
) -> list[dict[str, Any]]:
    cleaned = strip_useful_suffix(strip_copilot_quote(body))
    if has_no_open_issue_conclusion(cleaned):
        return []
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 1600,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Split a Copilot pull request review digest into actionable issues. "
                    "Return JSON only with key issues. Each issue has severity, title, problem, "
                    "files_hint, validation_hint. Use severities S0/S1/S2. Return [] if no open issue."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "pr": pr,
                        "comment_ref": comment_ref,
                        "digest": cleaned,
                        "severity_mapping": {
                            "S0": "P0/P1/high risk/blocking/no-go/阻塞/阻断/高风险",
                            "S1": "P2/medium risk/中风险/中级风险",
                            "S2": "P3/low risk/informational/低风险/信息性",
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek digest API HTTP {exc.code}: {body_text[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"DeepSeek digest API request failed: {exc}") from exc

    choices = response_payload.get("choices") if isinstance(response_payload, dict) else None
    if not choices:
        raise RuntimeError("DeepSeek digest response did not include choices")
    content = choices[0].get("message", {}).get("content", "")
    parsed = _json_from_text(str(content))
    raw_issues = parsed.get("issues", [])
    if not isinstance(raw_issues, list):
        raise RuntimeError("DeepSeek digest response `issues` must be a list")

    source = f"pr:{pr}:copilot_digest:{comment_ref}:deepseek"
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_issues, start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "Copilot review issue")
        problem = str(item.get("problem") or title)
        severity = str(item.get("severity") or infer_review_severity(title + "\n" + problem)).upper()
        if severity not in {"S0", "S1", "S2"}:
            severity = infer_review_severity(severity + "\n" + title + "\n" + problem)
        files = item.get("files_hint", [])
        if not isinstance(files, list):
            files = []
        validation = item.get("validation_hint", [])
        if not isinstance(validation, list):
            validation = []
        out.append(
            {
                "id": f"PR{pr}-COPILOT-{_stable_short_id(comment_ref)}-{idx}",
                "source": source,
                "kind": "copilot_digest_comment",
                "severity": severity,
                "title": title[:140],
                "problem": strip_useful_suffix(problem),
                "expected_fix": extract_expected_fix(problem),
                "files_hint": sorted({str(path) for path in files}) or _extract_files(title + "\n" + problem),
                "validation_hint": [str(cmd).strip() for cmd in validation if str(cmd).strip()]
                or [cmd.strip() for cmd in COMMAND_RE.findall(problem)],
                "status": "open",
            }
        )
    return out


def parse_copilot_digest(
    body: str,
    pr: int,
    comment_ref: str,
    *,
    digest_engine: str | None = None,
    deepseek_api_key: str | None = None,
    deepseek_model: str = "deepseek-v4-flash",
    deepseek_base_url: str = "https://api.deepseek.com",
    deepseek_timeout: float = 20.0,
    require_deepseek: bool = False,
) -> list[dict[str, Any]]:
    digest_engine = str(digest_engine or configured_env("AGENT_REVIEW_DIGEST_ENGINE", "deterministic")).lower()
    if digest_engine == "deepseek" and not deepseek_api_key:
        deepseek_api_key = configured_env("DEEPSEEK_API_KEY")
    if digest_engine == "deepseek" and deepseek_api_key:
        try:
            return parse_copilot_digest_comment_with_deepseek(
                body,
                pr,
                comment_ref,
                api_key=deepseek_api_key,
                model=deepseek_model,
                base_url=deepseek_base_url,
                timeout=deepseek_timeout,
            )
        except RuntimeError:
            if require_deepseek:
                raise
    elif digest_engine == "deepseek" and require_deepseek:
        raise RuntimeError("DeepSeek digest parsing requires DEEPSEEK_API_KEY")
    return parse_copilot_digest_comment(body, pr, comment_ref)


def issues_from_payload(
    payload: dict[str, Any],
    pr: int,
    *,
    include_pr_comments: bool = False,
    digest_engine: str | None = None,
    deepseek_api_key: str | None = None,
    deepseek_model: str = "deepseek-v4-flash",
    deepseek_base_url: str = "https://api.deepseek.com",
    deepseek_timeout: float = 20.0,
    require_deepseek: bool = False,
) -> list[dict[str, Any]]:
    pr_data = payload.get("data", {}).get("repository", {}).get("pullRequest", {})
    issues: list[dict[str, Any]] = []
    for thread in pr_data.get("reviewThreads", {}).get("nodes", []):
        if not isinstance(thread, dict) or thread.get("isResolved"):
            continue
        comments = [comment for comment in thread.get("comments", {}).get("nodes", []) if isinstance(comment, dict)]
        if not comments:
            continue
        first = comments[0]
        body = strip_useful_suffix(str(first.get("body") or ""))
        path = str(first.get("path") or "")
        line = first.get("line")
        thread_id = str(thread.get("id") or "")
        issue = {
            "id": f"PR{pr}-{_short_thread_id(thread_id)}",
            "source": f"pr:{pr}:thread:{thread_id}",
            "kind": "inline_thread",
            "severity": infer_review_severity(body),
            "title": extract_review_title(body),
            "problem": body.strip(),
            "expected_fix": extract_expected_fix(body),
            "files_hint": [path] if path else [],
            "line_hint": line,
            "thread_id": thread_id,
            "validation_hint": [cmd.strip() for cmd in COMMAND_RE.findall(body)],
            "status": "open",
        }
        if thread.get("isOutdated"):
            issue["stale"] = True
            issue["outdated"] = True
        issues.append(issue)
    for idx, comment in enumerate(pr_data.get("comments", {}).get("nodes", []), start=1):
        if not isinstance(comment, dict) or not comment.get("body"):
            continue
        body = str(comment["body"])
        kind = classify_pr_comment(comment)
        if kind in {"trigger_comment", "human_fix_report"}:
            continue
        if kind == "plain_comment" and not include_pr_comments:
            continue
        comment_ref = str(comment.get("id") or comment.get("url") or idx)
        if kind == "copilot_digest_comment":
            issues.extend(
                parse_copilot_digest(
                    body,
                    pr,
                    comment_ref,
                    digest_engine=digest_engine,
                    deepseek_api_key=deepseek_api_key,
                    deepseek_model=deepseek_model,
                    deepseek_base_url=deepseek_base_url,
                    deepseek_timeout=deepseek_timeout,
                    require_deepseek=require_deepseek,
                )
            )
            continue
        if kind == "plain_actionable_comment" and not include_pr_comments and not is_actionable_pr_comment(body):
            continue
        created_at = str(comment.get("createdAt") or idx)
        issues.append(
            {
                "id": f"PR{pr}-COMMENT-{idx}",
                "source": f"pr:{pr}:comment:{created_at}",
                "kind": kind,
                "severity": infer_review_severity(body),
                "title": extract_review_title(body),
                "problem": strip_useful_suffix(body),
                "expected_fix": extract_expected_fix(body),
                "files_hint": [],
                "validation_hint": [cmd.strip() for cmd in COMMAND_RE.findall(body)],
                "status": "open",
            }
        )
    return issues


def ingest_review_inbox(
    pr: int,
    out_dir: Path,
    *,
    review_file: Path | None = None,
    include_pr_comments: bool = False,
    digest_engine: str | None = None,
    deepseek_api_key: str | None = None,
    deepseek_model: str = "deepseek-v4-flash",
    deepseek_base_url: str = "https://api.deepseek.com",
    deepseek_timeout: float = 20.0,
    require_deepseek: bool = False,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    issues: list[dict[str, Any]] = []
    fetch_error: str | None = None
    try:
        payload = fetch_pr_review_threads(pr)
        (out_dir / "pr_review_raw.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        text = review_text_from_payload(payload)
        (out_dir / "pr_review_inbox.md").write_text(text, encoding="utf-8")
        issues.extend(
            issues_from_payload(
                payload,
                pr,
                include_pr_comments=include_pr_comments,
                digest_engine=digest_engine,
                deepseek_api_key=deepseek_api_key,
                deepseek_model=deepseek_model,
                deepseek_base_url=deepseek_base_url,
                deepseek_timeout=deepseek_timeout,
                require_deepseek=require_deepseek,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI fallback should preserve clear failure text
        fetch_error = str(exc)
        if review_file is None:
            raise RuntimeError(f"failed to fetch PR review comments and no --review-file fallback was provided: {fetch_error}") from exc

    if review_file is not None:
        review_text = review_file.read_text(encoding="utf-8")
        if looks_like_copilot_digest(review_text):
            issues.extend(
                parse_copilot_digest(
                    review_text,
                    pr,
                    f"review-file:{review_file.as_posix()}",
                    digest_engine=digest_engine,
                    deepseek_api_key=deepseek_api_key,
                    deepseek_model=deepseek_model,
                    deepseek_base_url=deepseek_base_url,
                    deepseek_timeout=deepseek_timeout,
                    require_deepseek=require_deepseek,
                )
            )
        else:
            issues.extend(parse_text(review_text, source=f"review-file:{review_file.as_posix()}"))
    issues = dedupe_issues(issues)
    if fetch_error:
        (out_dir / "fetch_error.txt").write_text(fetch_error + "\n", encoding="utf-8")
    write_issue_board(issues, out_dir)
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch or ingest PR reviews into an issue board.")
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--include-pr-comments", action="store_true", help="Include all top-level PR comments, not only actionable comments.")
    parser.add_argument("--digest-engine", choices=("deterministic", "deepseek"), default=str(configured_env("AGENT_REVIEW_DIGEST_ENGINE", "deterministic")).lower())
    parser.add_argument("--deepseek-model", default=configured_env("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--deepseek-base-url", default=configured_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--deepseek-timeout", type=float, default=20.0)
    parser.add_argument("--require-deepseek", action="store_true")
    args = parser.parse_args(argv)
    try:
        issues = ingest_review_inbox(
            args.pr,
            args.out,
            review_file=args.review_file,
            include_pr_comments=args.include_pr_comments,
            digest_engine=args.digest_engine,
            deepseek_api_key=configured_env("DEEPSEEK_API_KEY"),
            deepseek_model=args.deepseek_model,
            deepseek_base_url=args.deepseek_base_url,
            deepseek_timeout=args.deepseek_timeout,
            require_deepseek=args.require_deepseek,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1
    print(json.dumps({"issue_count": len(issues), "issue_board": str((args.out / "issue_board.json").resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
