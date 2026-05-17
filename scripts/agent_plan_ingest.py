from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SECTION_RE = re.compile(r"(?im)^\s*(?:#{1,6}\s*)?(?P<label>S[0-2](?:-\d+)?|Blocking|No-Go|仍然阻断|需要补)\s*[:：\-]?\s*(?P<title>.*)$")
PATH_RE = re.compile(r"(?<![\w.-])(?:[\w.-]+/[\w./-]+|[\w.-]+\\[\w.\\/-]+|\w+\.py|\w+\.md|\w+\.yaml|\w+\.json)(?![\w.-])")
COMMAND_RE = re.compile(r"(?m)^\s*(?:python|pytest|git|gh)\s+[^\n]+")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def severity_from_label(label: str, body: str) -> str:
    value = label.upper()
    lowered = body.lower()
    if value.startswith("S0") or "blocking" in lowered or "no-go" in lowered or "阻断" in body:
        return "S0"
    if value.startswith("S1"):
        return "S1"
    if value.startswith("S2"):
        return "S2"
    return "S1"


def extract_expected_fix(body: str) -> str:
    lines = [line.strip("-* \t") for line in body.splitlines()]
    capture: list[str] = []
    for idx, line in enumerate(lines):
        if any(marker in line for marker in ("需要补", "需要改", "请", "Expected fix", "Fix", "修复")):
            capture.append(line)
            capture.extend(item for item in lines[idx + 1 : idx + 5] if item)
            break
    return "\n".join(capture[:5])


def extract_title(label: str, raw_title: str, body: str) -> str:
    title = raw_title.strip(" ：:-")
    if title:
        return title[:140]
    for line in body.splitlines():
        clean = line.strip(" #*-`")
        if clean and clean.lower() not in {"text", "python", "bash"}:
            return clean[:140]
    return label


def split_sections(text: str) -> list[tuple[str, str, str]]:
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [("Issue", "Unstructured plan item", text)] if text.strip() else []
    sections: list[tuple[str, str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections.append((match.group("label"), match.group("title"), text[start:end].strip()))
    return sections


def issue_id(title: str, problem: str) -> str:
    digest = hashlib.sha1(normalize(title + " " + problem).encode("utf-8")).hexdigest()[:8]
    return f"ISS-{digest}"


def parse_text(text: str, *, source: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for label, raw_title, body in split_sections(text):
        title = extract_title(label, raw_title, body)
        problem = body.strip()
        key = normalize(title + " " + problem[:400])
        if key in seen:
            continue
        seen.add(key)
        files = sorted({match.rstrip(".,;:)") for match in PATH_RE.findall(problem)})
        commands = [cmd.strip() for cmd in COMMAND_RE.findall(problem)]
        issue = {
            "id": issue_id(title, problem),
            "source": source,
            "severity": severity_from_label(label, problem),
            "title": title,
            "problem": problem,
            "expected_fix": extract_expected_fix(problem),
            "files_hint": files,
            "validation_hint": commands,
            "status": "open",
        }
        issues.append(issue)
    return issues


def dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for issue in issues:
        key = normalize(str(issue["title"]) + " " + str(issue["problem"])[:500])
        if key in seen:
            continue
        seen.add(key)
        out.append(issue)
    return out


def write_issue_board(issues: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "issue_board.json"
    md_path = out_dir / "issue_board.md"
    payload = {"issues": issues}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Issue Board", ""]
    for issue in issues:
        lines.extend(
            [
                f"## {issue['id']} [{issue['severity']}] {issue['title']}",
                "",
                f"- source: {issue['source']}",
                f"- status: {issue['status']}",
                f"- files_hint: {', '.join(issue['files_hint']) if issue['files_hint'] else 'none'}",
                f"- validation_hint: {', '.join(issue['validation_hint']) if issue['validation_hint'] else 'none'}",
                "",
                "Problem:",
                issue["problem"] or "(empty)",
                "",
                "Expected fix:",
                issue["expected_fix"] or "(not specified)",
                "",
            ]
        )
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return json_path, md_path


def ingest(plan: Path, out_dir: Path, *, review_file: Path | None = None, pr: str | None = None) -> list[dict[str, Any]]:
    issues = parse_text(plan.read_text(encoding="utf-8"), source=f"plan:{plan.as_posix()}")
    if review_file is not None:
        issues.extend(parse_text(review_file.read_text(encoding="utf-8"), source=f"review:{review_file.as_posix()}"))
    if pr:
        for issue in issues:
            issue["source"] = f"{issue['source']} pr:{pr}"
    issues = dedupe_issues(issues)
    write_issue_board(issues, out_dir)
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest a long human/AI plan into an issue board.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--pr")
    args = parser.parse_args(argv)

    issues = ingest(args.plan, args.out, review_file=args.review_file, pr=args.pr)
    print(json.dumps({"issue_count": len(issues), "issue_board": str((args.out / "issue_board.json").resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
