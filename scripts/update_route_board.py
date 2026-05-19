from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from hust_bci_er.audit.promotion import parse_key_value_markdown  # noqa: E402

BOARD_PATH = ROOT / "reports" / "route_board.md"
REGISTRY_PATH = ROOT / "reports" / "route_registry.yaml"
SUMMARY_REQUIRED_STATUSES = {"CANDIDATE", "PROMOTED", "REJECTED", "ARCHIVED"}
PENDING_AUDIT_PREFIXES = ("PENDING",)
PLACEHOLDER_TOKENS = ("placeholder", "pending remote", "pending real", "temporary")


def route_files() -> list[Path]:
    return sorted((ROOT / "configs" / "routes" / "models").glob("*.yaml"))


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"route config must be a mapping: {path}")
    return data


def route_registry() -> dict[str, dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        return {}
    data = load_yaml(REGISTRY_PATH)
    entries = data.get("routes") or []
    if not isinstance(entries, list):
        return {}
    return {
        str(item.get("route_id")): item
        for item in entries
        if isinstance(item, dict) and isinstance(item.get("route_id"), str)
    }


def model_name(model: Any) -> str:
    if isinstance(model, str):
        return model
    if isinstance(model, dict):
        return str(model.get("name") or "")
    return ""


def protocol_name(evaluation: Any) -> str:
    if isinstance(evaluation, str):
        return evaluation
    if isinstance(evaluation, dict):
        return str(evaluation.get("protocol") or "")
    return ""


def primary_metric(evaluation: Any) -> str:
    if isinstance(evaluation, dict):
        return str(evaluation.get("primary_metric") or "")
    return ""


def job_adapter(training: Any) -> str:
    if isinstance(training, dict):
        return str(training.get("job_adapter") or "")
    return ""


def is_placeholder_summary(fields: dict[str, str]) -> bool:
    audit_decision = str(fields.get("audit_decision", "")).upper()
    if audit_decision.startswith(PENDING_AUDIT_PREFIXES):
        return True
    metric_value = str(fields.get("primary_metric_value", "")).strip().upper()
    if metric_value in {"TBD", "PENDING"}:
        return True
    notes = " ".join(str(fields.get(key, "")) for key in ("risk notes", "decision", "completed_jobs")).lower()
    return any(token in notes for token in PLACEHOLDER_TOKENS)


def generate_board() -> str:
    registry = route_registry()
    lines = [
        "# Route Board",
        "",
        "generated_by: `python scripts/update_route_board.py`",
        "",
        "| route_id | owner | status | latest_gate | primary_metric | dataset | split | model | adapter | protocol | summary | blocker |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for path in route_files():
        data = load_yaml(path)
        route_id = str(data.get("route_id") or path.stem)
        status = str(data.get("status") or "")
        model = model_name(data.get("model"))
        adapter_name = job_adapter(data.get("training"))
        evaluation = data.get("evaluation")
        protocol = protocol_name(evaluation)
        metric = primary_metric(evaluation)
        dataset = str(data.get("dataset_version") or "")
        split = str(data.get("split_id") or "")
        summary = ROOT / "reports" / "route_summaries" / f"{route_id}_summary.md"
        fields = parse_key_value_markdown(summary) if summary.exists() else {}
        registry_entry = registry.get(route_id, {})
        owner = str(registry_entry.get("owner") or "")
        if summary.exists():
            summary_state = "pending_placeholder" if is_placeholder_summary(fields) else "present"
        elif status in SUMMARY_REQUIRED_STATUSES:
            summary_state = "missing_required"
        else:
            summary_state = "not_required"
        latest_gate = "" if summary_state == "pending_placeholder" else fields.get("gate", "")
        blocker = fields.get("risk notes", "") or str(registry_entry.get("blocker") or "")
        lines.append(f"| `{route_id}` | {owner} | {status} | {latest_gate} | `{metric}` | `{dataset}` | `{split}` | `{model}` | `{adapter_name}` | `{protocol}` | {summary_state} | {blocker} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    expected = generate_board()
    if args.check:
        actual = BOARD_PATH.read_text(encoding="utf-8") if BOARD_PATH.exists() else ""
        if actual != expected:
            print("Route board is not synchronized. Run: python scripts/update_route_board.py", file=sys.stderr)
            return 1
        print("Route board is synchronized.")
        return 0

    BOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOARD_PATH.write_text(expected, encoding="utf-8")
    print(f"Updated {BOARD_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
