from __future__ import annotations

import argparse
import csv
import re
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
REMOTE_PROTOCOL_LEDGER_ROOT = ROOT / "reports" / "model_deep_dive"
SUMMARY_REQUIRED_STATUSES = {"CANDIDATE", "PROMOTED", "REJECTED", "ARCHIVED"}
PENDING_AUDIT_PREFIXES = ("PENDING",)
PLACEHOLDER_TOKENS = ("placeholder", "pending remote", "pending real", "temporary")
SCORE_VALUE_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+)")
SCORE_HEADER_RE = re.compile(
    r"(?:\bexact_single_crop_expected_ba\b|\bba\b|\bmetric(?:[_ -]?(?:value|mean|std|min|max))?\b|\bscore\b|\bdep/hc(?:\s+ratio)?\b)",
    flags=re.IGNORECASE,
)
SCORE_BULLET_VALUE_RE = re.compile(
    r"(?:\bexact_single_crop_expected_ba\b|\b(?:exact|mean|final|eval|crop\d+)\s+ba\b|\bba\b|"
    r"\bmetric(?:[_ -]?(?:value|mean|std|min|max))?\b|\bscore\b|\bdep/hc(?:\s+ratio)?\b)"
    r"\s*(?::|=|\s+)\s*[-+]?(?:\d+\.\d+|\d+)",
    flags=re.IGNORECASE,
)


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
    remote_score_items = remote_protocol_score_items_by_route()
    score_rows: list[str] = []
    lines = [
        "# Route Board",
        "",
        "generated_by: `python scripts/update_route_board.py`",
        "",
        "| route_id | owner | board_state | config_status | summary_status | audit_decision | latest_gate | primary_metric | dataset | split | model | adapter | protocol | summary | blocker |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
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
        summary_status = fields.get("route_status", "")
        audit_decision = fields.get("audit_decision", "")
        has_remote_scores = bool(remote_score_items.get(route_id))
        state = board_state(status, summary_state, fields, has_remote_scores=has_remote_scores)
        blocker = route_blocker(summary_state, fields, registry_entry, has_remote_scores=has_remote_scores)
        lines.append(
            f"| `{route_id}` | {owner} | {state} | {status} | {summary_status} | {audit_decision} | {latest_gate} | `{metric}` | `{dataset}` | `{split}` | `{model}` | `{adapter_name}` | `{protocol}` | {summary_state} | {blocker} |"
        )
        score_items = score_ledger_items(summary, fields) if summary_state == "present" and summary.exists() else []
        score_items.extend(remote_score_items.get(route_id, []))
        score_items = unique_preserve_order(score_items)
        if score_items:
            score_rows.append(f"| `{route_id}` | {state} | {summary_state} | {format_score_items(score_items)} |")
    lines.extend(
        [
            "",
            "## Route Score Ledger",
            "",
            "This section lists committed score summaries from route summaries and remote protocol score ledgers under `reports/model_deep_dive/`. It does not scan raw `outputs/` directories and must not be treated as route-status promotion evidence by itself.",
            "",
            "| route_id | board_state | summary | committed score summaries |",
            "|---|---|---|---|",
        ]
    )
    lines.extend(score_rows)
    return "\n".join(lines) + "\n"


def board_state(config_status: str, summary_state: str, fields: dict[str, str], *, has_remote_scores: bool = False) -> str:
    audit_decision = str(fields.get("audit_decision", "")).strip()
    audit_upper = audit_decision.upper()
    if audit_upper.startswith("BLOCKED"):
        return "BLOCKED"
    if summary_state == "missing_required":
        if has_remote_scores:
            return "REMOTE_SCORES_PRESENT"
        return "MISSING_SUMMARY"
    if summary_state == "pending_placeholder":
        if has_remote_scores:
            return "REMOTE_SCORES_PRESENT"
        return "PENDING"
    if "DIAGNOSTIC_ONLY" in audit_upper:
        return "DIAGNOSTIC_ONLY"
    if "PASS" in audit_upper:
        return "AUDIT_PASS"
    if audit_upper.startswith(PENDING_AUDIT_PREFIXES):
        return "PENDING"
    return str(fields.get("route_status") or config_status or "")


def route_blocker(summary_state: str, fields: dict[str, str], registry_entry: dict[str, Any], *, has_remote_scores: bool = False) -> str:
    audit_decision = str(fields.get("audit_decision", "")).strip()
    audit_upper = audit_decision.upper()
    decision = str(fields.get("decision", "")).strip()
    risk_notes = str(fields.get("risk notes", "")).strip()
    registry_blocker = str(registry_entry.get("blocker") or "").strip()
    if audit_upper.startswith("BLOCKED"):
        return clean_score_text(decision or risk_notes or registry_blocker or audit_decision)
    if summary_state == "missing_required":
        if has_remote_scores:
            return "summary_refresh_pending"
        return "missing_required_summary"
    if summary_state == "pending_placeholder":
        if has_remote_scores:
            return "summary_refresh_pending"
        return audit_decision or "pending_placeholder"
    if "DIAGNOSTIC_ONLY" in audit_upper or "diagnostic" in decision.lower():
        return "diagnostic_only"
    if "PASS" in audit_upper:
        return ""
    return clean_score_text(registry_blocker or risk_notes)


def remote_protocol_score_items_by_route() -> dict[str, list[str]]:
    by_route: dict[str, list[str]] = {}
    for path in sorted(REMOTE_PROTOCOL_LEDGER_ROOT.glob("remote_protocol_score_ledger_*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = normalize_csv_row(row)
                route_id = str(row.get("route_id") or "").strip()
                if not route_id:
                    continue
                item = remote_protocol_score_item(row, path)
                if item:
                    by_route.setdefault(route_id, []).append(item)
    return {route: unique_preserve_order(items) for route, items in by_route.items()}


def normalize_csv_row(row: dict[str, str]) -> dict[str, str]:
    return {
        str(key).strip(): str(value).strip()
        for key, value in row.items()
        if key is not None
    }


def remote_protocol_score_item(row: dict[str, str], path: Path) -> str:
    machine = clean_score_text(row.get("machine", ""))
    protocol = protocol_short_name(row.get("protocol", ""))
    run_id = clean_score_text(row.get("run_id", ""))
    mean = clean_float_text(row.get("mean", ""))
    std = clean_float_text(row.get("std", ""))
    min_value = clean_float_text(row.get("min", ""))
    max_value = clean_float_text(row.get("max", ""))
    n_jobs = clean_score_text(row.get("board_n_jobs", "") or row.get("n_jobs", ""))
    source = path.name.removesuffix(".csv")
    if not (machine and protocol and mean):
        return ""
    return (
        f"{source}:{machine}:{protocol}:{run_id}:"
        f"mean={mean},std={std},min={min_value},max={max_value},n={n_jobs}"
    )


def protocol_short_name(value: str | None) -> str:
    protocol = str(value or "").strip()
    if protocol.startswith("p1"):
        return "P1"
    if protocol.startswith("p2"):
        return "P2"
    if protocol.startswith("p3"):
        return "P3"
    return clean_score_text(protocol)


def clean_float_text(value: str | None) -> str:
    try:
        return f"{float(str(value).strip()):.4f}"
    except (TypeError, ValueError):
        return clean_score_text(str(value or ""))


def score_ledger_items(summary_path: Path, fields: dict[str, str]) -> list[str]:
    text = summary_path.read_text(encoding="utf-8")
    items: list[str] = []
    for key, value in fields.items():
        if is_score_key(key) and has_number(value):
            items.append(f"{key}={clean_score_text(value)}")
    items.extend(markdown_table_score_items(text))
    items.extend(markdown_bullet_score_items(text))
    return unique_preserve_order(items)


def is_score_key(key: str) -> bool:
    lowered = key.lower()
    if lowered == "primary_metric":
        return False
    return "metric" in lowered or lowered.startswith("primary_metric_value")


def markdown_table_score_items(text: str) -> list[str]:
    lines = text.splitlines()
    items: list[str] = []
    idx = 0
    while idx < len(lines):
        if not lines[idx].lstrip().startswith("|"):
            idx += 1
            continue
        header = split_markdown_row(lines[idx])
        if not header or not any(is_score_header(cell) for cell in header):
            idx += 1
            continue
        if idx + 1 >= len(lines) or not is_markdown_separator(lines[idx + 1]):
            idx += 1
            continue
        score_indices = [pos for pos, cell in enumerate(header) if is_score_header(cell)]
        label_indices = [pos for pos in range(len(header)) if pos not in score_indices]
        idx += 2
        while idx < len(lines) and lines[idx].lstrip().startswith("|"):
            cells = split_markdown_row(lines[idx])
            for score_pos in score_indices:
                if score_pos >= len(cells) or not has_number(cells[score_pos]):
                    continue
                row_parts = [
                    f"{clean_header(header[pos])}={clean_score_text(cells[pos])}"
                    for pos in label_indices
                    if pos < len(cells) and cells[pos].strip()
                ]
                row_context = ", ".join(row_parts) if row_parts else "row"
                items.append(f"{row_context}: {clean_header(header[score_pos])}={clean_score_text(cells[score_pos])}")
            idx += 1
        continue
    return items


def markdown_bullet_score_items(text: str) -> list[str]:
    items: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped.startswith("- "):
            continue
        item = stripped[2:].rstrip(".")
        if not SCORE_BULLET_VALUE_RE.search(item):
            continue
        items.append(clean_score_text(item))
    return items


def split_markdown_row(row: str) -> list[str]:
    stripped = row.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def is_markdown_separator(row: str) -> bool:
    cells = split_markdown_row(row)
    return bool(cells) and all(set(cell.replace(":", "").strip()) <= {"-"} for cell in cells)


def is_score_header(value: str) -> bool:
    lowered = clean_header(value).lower()
    return bool(SCORE_HEADER_RE.search(lowered))


def clean_header(value: str) -> str:
    return value.strip().strip("`").replace(" ", "_")


def clean_score_text(value: str) -> str:
    return " ".join(str(value).strip().split()).replace("|", "\\|")


def has_number(value: str) -> bool:
    return bool(SCORE_VALUE_RE.search(str(value)))


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def format_score_items(items: list[str]) -> str:
    return "<br>".join(format_inline_code(item) for item in items)


def format_inline_code(value: str) -> str:
    max_backtick_run = max((len(match.group(0)) for match in re.finditer(r"`+", value)), default=0)
    delimiter = "`" * (max_backtick_run + 1)
    if value.startswith("`") or value.endswith("`"):
        return f"{delimiter} {value} {delimiter}"
    return f"{delimiter}{value}{delimiter}"


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
