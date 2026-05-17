from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    from scripts.agent_plan_ingest import COMMAND_RE, dedupe_issues, extract_expected_fix, parse_text, write_issue_board
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from agent_plan_ingest import COMMAND_RE, dedupe_issues, extract_expected_fix, parse_text, write_issue_board


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
        ["gh", "repo", "view", "--json", "owner,name"],
        text=True,
        capture_output=True,
        check=True,
    )
    repo_data = json.loads(repo.stdout)
    owner = repo_data["owner"]["login"] if isinstance(repo_data.get("owner"), dict) else repo_data["owner"]
    name = repo_data["name"]
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


def infer_review_severity(body: str) -> str:
    lowered = body.lower()
    if any(marker in lowered for marker in ("[p0]", "p0:", "[p1]", "p1:", "s0", "blocking", "no-go", "阻断")):
        return "S0"
    if any(marker in lowered for marker in ("[p2]", "p2:", "s1")):
        return "S1"
    if any(marker in lowered for marker in ("[p3]", "p3:", "s2")):
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


def issues_from_payload(payload: dict[str, Any], pr: int, *, include_pr_comments: bool = False) -> list[dict[str, Any]]:
    pr_data = payload.get("data", {}).get("repository", {}).get("pullRequest", {})
    issues: list[dict[str, Any]] = []
    for thread in pr_data.get("reviewThreads", {}).get("nodes", []):
        if not isinstance(thread, dict) or thread.get("isResolved"):
            continue
        comments = [comment for comment in thread.get("comments", {}).get("nodes", []) if isinstance(comment, dict)]
        if not comments:
            continue
        first = comments[0]
        body = str(first.get("body") or "")
        path = str(first.get("path") or "")
        line = first.get("line")
        thread_id = str(thread.get("id") or "")
        issues.append(
            {
                "id": f"PR{pr}-{_short_thread_id(thread_id)}",
                "source": f"pr:{pr}:thread:{thread_id}",
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
        )
    for idx, comment in enumerate(pr_data.get("comments", {}).get("nodes", []), start=1):
        if not isinstance(comment, dict) or not comment.get("body"):
            continue
        body = str(comment["body"])
        if not include_pr_comments and not is_actionable_pr_comment(body):
            continue
        created_at = str(comment.get("createdAt") or idx)
        issues.append(
            {
                "id": f"PR{pr}-COMMENT-{idx}",
                "source": f"pr:{pr}:comment:{created_at}",
                "severity": infer_review_severity(body),
                "title": extract_review_title(body),
                "problem": body.strip(),
                "expected_fix": extract_expected_fix(body),
                "files_hint": [],
                "validation_hint": [cmd.strip() for cmd in COMMAND_RE.findall(body)],
                "status": "open",
            }
        )
    return issues


def ingest_review_inbox(pr: int, out_dir: Path, *, review_file: Path | None = None, include_pr_comments: bool = False) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    issues: list[dict[str, Any]] = []
    fetch_error: str | None = None
    try:
        payload = fetch_pr_review_threads(pr)
        (out_dir / "pr_review_raw.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        text = review_text_from_payload(payload)
        (out_dir / "pr_review_inbox.md").write_text(text, encoding="utf-8")
        issues.extend(issues_from_payload(payload, pr, include_pr_comments=include_pr_comments))
    except Exception as exc:  # noqa: BLE001 - CLI fallback should preserve clear failure text
        fetch_error = str(exc)
        if review_file is None:
            raise RuntimeError(f"failed to fetch PR review comments and no --review-file fallback was provided: {fetch_error}") from exc

    if review_file is not None:
        issues.extend(parse_text(review_file.read_text(encoding="utf-8"), source=f"review-file:{review_file.as_posix()}"))
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
    args = parser.parse_args(argv)
    try:
        issues = ingest_review_inbox(args.pr, args.out, review_file=args.review_file, include_pr_comments=args.include_pr_comments)
    except RuntimeError as exc:
        print(str(exc))
        return 1
    print(json.dumps({"issue_count": len(issues), "issue_board": str((args.out / "issue_board.json").resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
