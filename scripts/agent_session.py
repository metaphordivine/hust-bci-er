from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from scripts.agent_context import context_for_task
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from agent_context import context_for_task


ROOT = Path(__file__).resolve().parents[1]


def default_session_dir(task_family: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = task_family.replace("/", "-").replace(" ", "-")
    return ROOT / "scratch" / "agent_sessions" / f"{stamp}_{safe}"


def session_paths(session_dir: Path) -> tuple[Path, Path]:
    return session_dir / "session.json", session_dir / "session.md"


def load_session(session_dir: Path) -> dict[str, Any]:
    json_path, _ = session_paths(session_dir)
    if not json_path.exists():
        raise FileNotFoundError(f"session.json not found: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def write_session(session_dir: Path, data: dict[str, Any]) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = session_paths(session_dir)
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Agent Session",
        "",
        f"- task_family: {data.get('task_family', '')}",
        f"- intake_mode: {data.get('intake_mode', '')}",
        f"- selected_context_pack: {data.get('selected_context_pack', '')}",
        f"- issue_board_path: {data.get('issue_board_path', '')}",
        f"- validation_status: {data.get('validation_status', '')}",
        f"- next_action: {data.get('next_action', '')}",
        "",
        "## Files Read",
    ]
    files_read = data.get("files_read") or []
    if not files_read:
        lines.append("- none")
    for item in files_read:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Commands Run",
        ]
    )
    commands_run = data.get("commands_run") or []
    if not commands_run:
        lines.append("- none")
    for item in commands_run:
        if isinstance(item, dict):
            lines.append(f"- {item.get('timestamp', '')}: {item.get('command', '')}")
        else:
            lines.append(f"- {item}")
    lines.extend(
        [
        "",
        "## Issues",
        ]
    )
    issues = data.get("issues") or {}
    if not issues:
        lines.append("- none")
    for issue_id, issue in sorted(issues.items()):
        lines.append(f"- {issue_id}: {issue.get('status')} - {issue.get('message', '')}")
    lines.extend(["", "## Notes"])
    notes = data.get("notes") or []
    if not notes:
        lines.append("- none")
    for note in notes:
        lines.append(f"- {note['timestamp']}: {note['message']}")
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def init_session(task_family: str, session_dir: Path | None, *, intake_mode: str | None = None) -> Path:
    context = context_for_task(task_family)
    out_dir = session_dir or default_session_dir(task_family)
    data = {
        "task_family": context["task_family"],
        "intake_mode": intake_mode or "",
        "selected_context_pack": context["context_pack_path"],
        "issue_board_path": str((out_dir / "issue_board.json").as_posix()),
        "files_read": [],
        "commands_run": [],
        "issues": {},
        "validation_status": "not_run",
        "next_action": "read selected context pack",
        "notes": [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_session(out_dir, data)
    return out_dir


def add_note(session_dir: Path, message: str) -> None:
    data = load_session(session_dir)
    data.setdefault("notes", []).append({"timestamp": datetime.now().isoformat(timespec="seconds"), "message": message})
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_session(session_dir, data)


def update_issue(session_dir: Path, issue_id: str, status: str, message: str) -> None:
    data = load_session(session_dir)
    data.setdefault("issues", {})[issue_id] = {
        "status": status,
        "message": message,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    open_items = [item for item in data["issues"].values() if item.get("status") == "open"]
    data["next_action"] = "fix open issues" if open_items else "run validation"
    write_session(session_dir, data)


def record_file_read(session_dir: Path, file_path: str) -> None:
    data = load_session(session_dir)
    files = data.setdefault("files_read", [])
    if file_path not in files:
        files.append(file_path)
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_session(session_dir, data)


def record_command(session_dir: Path, command: str) -> None:
    data = load_session(session_dir)
    data.setdefault("commands_run", []).append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "command": command,
        }
    )
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_session(session_dir, data)


def update_validation(session_dir: Path, status: str, message: str) -> None:
    data = load_session(session_dir)
    data["validation_status"] = status
    data["validation_message"] = message
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_session(session_dir, data)


def update_next_action(session_dir: Path, message: str) -> None:
    data = load_session(session_dir)
    data["next_action"] = message
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_session(session_dir, data)


def print_status(session_dir: Path) -> None:
    data = load_session(session_dir)
    print(f"session_dir: {session_dir}")
    print(f"task_family: {data.get('task_family', '')}")
    print(f"intake_mode: {data.get('intake_mode', '')}")
    print(f"selected_context_pack: {data.get('selected_context_pack', '')}")
    print(f"validation_status: {data.get('validation_status', '')}")
    print(f"next_action: {data.get('next_action', '')}")
    issues = data.get("issues") or {}
    print(f"issues: {len(issues)}")
    print(f"files_read: {len(data.get('files_read') or [])}")
    print(f"commands_run: {len(data.get('commands_run') or [])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Maintain a minimal agent session ledger.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--task", required=True)
    init.add_argument("--session-dir", type=Path)
    init.add_argument("--intake-mode")

    note = sub.add_parser("note")
    note.add_argument("--session-dir", type=Path, required=True)
    note.add_argument("--message", required=True)

    issue = sub.add_parser("issue")
    issue.add_argument("--session-dir", type=Path, required=True)
    issue.add_argument("--issue", required=True)
    issue.add_argument("--status", choices=["open", "fixed", "deferred"], required=True)
    issue.add_argument("--message", required=True)

    read = sub.add_parser("read")
    read.add_argument("--session-dir", type=Path, required=True)
    read.add_argument("--file", required=True)

    command = sub.add_parser("command")
    command.add_argument("--session-dir", type=Path, required=True)
    command.add_argument("--command", dest="command_text", required=True)

    validation = sub.add_parser("validation")
    validation.add_argument("--session-dir", type=Path, required=True)
    validation.add_argument("--status", choices=["not_run", "pass", "fail"], required=True)
    validation.add_argument("--message", required=True)

    next_action = sub.add_parser("next")
    next_action.add_argument("--session-dir", type=Path, required=True)
    next_action.add_argument("--message", required=True)

    status = sub.add_parser("status")
    status.add_argument("--session-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "init":
        out_dir = init_session(args.task, args.session_dir, intake_mode=args.intake_mode)
        print(out_dir.as_posix())
    elif args.command == "note":
        add_note(args.session_dir, args.message)
    elif args.command == "issue":
        update_issue(args.session_dir, args.issue, args.status, args.message)
    elif args.command == "read":
        record_file_read(args.session_dir, args.file)
    elif args.command == "command":
        record_command(args.session_dir, args.command_text)
    elif args.command == "validation":
        update_validation(args.session_dir, args.status, args.message)
    elif args.command == "next":
        update_next_action(args.session_dir, args.message)
    elif args.command == "status":
        print_status(args.session_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
