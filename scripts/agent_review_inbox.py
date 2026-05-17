from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    from scripts.agent_plan_ingest import dedupe_issues, parse_text, write_issue_board
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from agent_plan_ingest import dedupe_issues, parse_text, write_issue_board


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


def ingest_review_inbox(pr: int, out_dir: Path, *, review_file: Path | None = None) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    issues: list[dict[str, Any]] = []
    fetch_error: str | None = None
    try:
        payload = fetch_pr_review_threads(pr)
        (out_dir / "pr_review_raw.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        text = review_text_from_payload(payload)
        (out_dir / "pr_review_inbox.md").write_text(text, encoding="utf-8")
        issues.extend(parse_text(text, source=f"pr:{pr}:reviewThreads"))
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
    args = parser.parse_args(argv)
    try:
        issues = ingest_review_inbox(args.pr, args.out, review_file=args.review_file)
    except RuntimeError as exc:
        print(str(exc))
        return 1
    print(json.dumps({"issue_count": len(issues), "issue_board": str((args.out / "issue_board.json").resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
