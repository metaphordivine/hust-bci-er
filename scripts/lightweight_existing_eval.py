from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RouteEvidence:
    route_id: str
    model: str = ""
    family: str = ""
    status: str = ""
    summary_metric: float | None = None
    protocol_means: dict[str, float] = field(default_factory=dict)
    protocol_jobs: dict[str, int] = field(default_factory=dict)
    protocol_sources: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProtocolBoardRow:
    route_id: str
    protocol: str
    mean: float
    n_jobs: int
    path: Path
    audit_status: str
    audit_mtime_ns: int


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _read_key_value_markdown(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip().lstrip("-").strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip().strip("`")
    return fields


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _model_name(model: Any) -> str:
    if isinstance(model, str):
        return model
    if isinstance(model, dict):
        return str(model.get("name") or "")
    return ""


def model_family(route_id: str, model: str) -> str:
    key = f"{route_id} {model}".lower()
    if "score_fusion" in key or "fusion" in key or "average" in key:
        return "fusion"
    if "cbramod" in key:
        return "cbramod"
    if "fbstcnet" in key or "fbcnet" in key:
        return "filterbank_cnn"
    if "dgcnn" in key or "lggnet" in key or "graph" in key:
        return "graph"
    if "deformer" in key or "conformer" in key or "transformer" in key:
        return "attention_hybrid"
    if "shallow" in key or "eegnet" in key:
        return "compact_cnn"
    return "other"


def load_route_metadata() -> dict[str, RouteEvidence]:
    out: dict[str, RouteEvidence] = {}
    for path in sorted((ROOT / "configs" / "routes" / "models").glob("*.yaml")):
        data = _read_yaml(path)
        route_id = str(data.get("route_id") or path.stem)
        model = _model_name(data.get("model"))
        out[route_id] = RouteEvidence(
            route_id=route_id,
            model=model,
            family=model_family(route_id, model),
            status=str(data.get("status") or ""),
        )
    return out


def merge_route_summaries(evidence: dict[str, RouteEvidence], summary_dir: Path) -> None:
    if not summary_dir.exists():
        return
    for path in sorted(summary_dir.glob("*_summary.md")):
        route_id = path.name.removesuffix("_summary.md")
        item = evidence.setdefault(route_id, RouteEvidence(route_id=route_id))
        fields = _read_key_value_markdown(path)
        item.status = item.status or fields.get("route_status", "")
        if fields.get("audit_decision", "").upper() != "PASS":
            continue
        item.summary_metric = _parse_float(fields.get("primary_metric_value"))
        for protocol, key in (("P1", "aggregate_metric_mean"), ("P2", "p2_metric_mean"), ("P3", "p3_metric_mean")):
            value = _parse_float(fields.get(key))
            if value is not None:
                item.protocol_means.setdefault(protocol, value)
                item.protocol_sources.setdefault(protocol, "route_summary")


def _protocol_from_board(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("protocol1"):
        return "P1"
    if name.startswith("protocol2"):
        return "P2"
    if name.startswith("protocol3"):
        return "P3"
    return "PX"


def _audit_path_for_board(path: Path) -> Path:
    return path.with_name(path.name.replace("_board.csv", "_audit.json"))


def _board_audit(path: Path) -> tuple[str, int]:
    audit_path = _audit_path_for_board(path)
    if not audit_path.exists():
        return "", 0
    try:
        data = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "INVALID", audit_path.stat().st_mtime_ns
    if not isinstance(data, dict):
        return "INVALID", audit_path.stat().st_mtime_ns
    return str(data.get("status") or "").upper(), audit_path.stat().st_mtime_ns


def _read_protocol_board_rows(path: Path, *, require_complete_audit: bool) -> list[ProtocolBoardRow]:
    if not path.exists():
        return []
    status, audit_mtime_ns = _board_audit(path)
    if require_complete_audit and status != "COMPLETE":
        return []
    if status and status != "COMPLETE":
        return []
    protocol = _protocol_from_board(path)
    rows: list[ProtocolBoardRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            route_id = str(row.get("route_id") or "").strip()
            mean = _parse_float(row.get("mean"))
            if not route_id or mean is None:
                continue
            try:
                n_jobs = int(float(row.get("n_jobs") or 0))
            except ValueError:
                n_jobs = 0
            rows.append(
                ProtocolBoardRow(
                    route_id=route_id,
                    protocol=protocol,
                    mean=mean,
                    n_jobs=n_jobs,
                    path=path,
                    audit_status=status,
                    audit_mtime_ns=audit_mtime_ns or path.stat().st_mtime_ns,
                )
            )
    return rows


def _select_protocol_board_rows(rows: list[ProtocolBoardRow]) -> list[ProtocolBoardRow]:
    selected: dict[tuple[str, str], ProtocolBoardRow] = {}
    for row in rows:
        key = (row.route_id, row.protocol)
        current = selected.get(key)
        if current is None:
            selected[key] = row
            continue
        row_rank = (row.audit_mtime_ns, row.path.as_posix())
        current_rank = (current.audit_mtime_ns, current.path.as_posix())
        if row_rank > current_rank:
            selected[key] = row
    return list(selected.values())


def merge_protocol_boards(evidence: dict[str, RouteEvidence], board_paths: list[Path], *, require_complete_audit: bool) -> None:
    rows: list[ProtocolBoardRow] = []
    for path in board_paths:
        rows.extend(_read_protocol_board_rows(path, require_complete_audit=require_complete_audit))
    for row in _select_protocol_board_rows(rows):
        item = evidence.setdefault(row.route_id, RouteEvidence(route_id=row.route_id))
        item.protocol_means[row.protocol] = row.mean
        item.protocol_sources[row.protocol] = "protocol_board"
        item.protocol_jobs[row.protocol] = row.n_jobs


def find_protocol_boards(run_roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in run_roots:
        if root.exists():
            paths.extend(sorted(root.rglob("protocol*_board.csv")))
    return paths


def default_board_dir() -> Path:
    return ROOT / "reports" / "model_deep_dive" / "lightweight_existing_eval_boards"


def default_protocol_boards() -> list[Path]:
    default_dir = default_board_dir()
    if not default_dir.exists():
        return []
    return sorted(default_dir.glob("protocol*_board.csv"))


def ranked(evidence: dict[str, RouteEvidence], protocol: str) -> list[RouteEvidence]:
    rows = [item for item in evidence.values() if protocol in item.protocol_means]
    return sorted(rows, key=lambda item: item.protocol_means[protocol], reverse=True)


def family_leaders(evidence: dict[str, RouteEvidence], protocol: str) -> list[RouteEvidence]:
    best: dict[str, RouteEvidence] = {}
    for item in ranked(evidence, protocol):
        if item.family not in best:
            best[item.family] = item
    return sorted(best.values(), key=lambda item: item.protocol_means[protocol], reverse=True)


def coverage_line(evidence: dict[str, RouteEvidence], protocol: str) -> str:
    rows = ranked(evidence, protocol)
    board_count = sum(1 for item in rows if item.protocol_sources.get(protocol) == "protocol_board")
    summary_count = sum(1 for item in rows if item.protocol_sources.get(protocol) == "route_summary")
    return (
        f"- {protocol}: {len(rows)} routes with any evidence "
        f"({board_count} protocol-board, {summary_count} route-summary-only)"
    )


def _format_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _table(rows: list[list[str]], headers: list[str]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def render_markdown(evidence: dict[str, RouteEvidence], *, top_k: int) -> str:
    lines = [
        "# Lightweight Existing Evidence Evaluation",
        "",
        "This report is generated from committed route summaries, committed compact protocol board snapshots, and explicitly provided COMPLETE protocol boards. It does not read raw EEG data, train models, or promote routes.",
        "",
        "## Coverage",
        "",
    ]
    for protocol in ("P1", "P2", "P3"):
        lines.append(coverage_line(evidence, protocol))
    lines.extend(["", "## Protocol Leaders", ""])
    for protocol in ("P1", "P2", "P3"):
        rows: list[list[str]] = []
        for item in ranked(evidence, protocol)[:top_k]:
            rows.append(
                [
                    f"`{item.route_id}`",
                    item.family,
                    item.model,
                    item.status,
                    _format_metric(item.protocol_means.get(protocol)),
                    str(item.protocol_jobs.get(protocol, "")),
                    item.protocol_sources.get(protocol, ""),
                ]
            )
        lines.extend([f"### {protocol}", ""])
        lines.extend(_table(rows, ["Route", "Family", "Model", "Status", "Mean", "Jobs", "Source"]))
        lines.append("")
    lines.extend(["## Family Leaders By P3", ""])
    rows = [
        [f"`{item.route_id}`", item.family, item.model, _format_metric(item.protocol_means.get("P3"))]
        for item in family_leaders(evidence, "P3")[:top_k]
    ]
    lines.extend(_table(rows, ["Route", "Family", "Model", "P3 Mean"]))
    lines.extend(["", "## P3 Minus P1 Gap", ""])
    gap_rows: list[list[str]] = []
    for item in sorted(evidence.values(), key=lambda x: (x.protocol_means.get("P3", -1.0)), reverse=True):
        if "P1" not in item.protocol_means or "P3" not in item.protocol_means:
            continue
        p1 = item.protocol_means["P1"]
        p3 = item.protocol_means["P3"]
        gap_rows.append([f"`{item.route_id}`", item.family, _format_metric(p1), _format_metric(p3), f"{p3 - p1:+.4f}"])
        if len(gap_rows) >= top_k:
            break
    lines.extend(_table(gap_rows, ["Route", "Family", "P1", "P3", "P3-P1"]))
    lines.extend(
        [
            "",
            "## Use",
            "",
            "- Treat this as triage evidence only; it is not a route promotion artifact.",
            "- Prefer candidates that combine high P3 score with family diversity.",
            "- Large negative P3-P1 gaps are follow-up diagnostics for selection robustness or cohort/session shift.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_report(
    *,
    summary_dir: Path,
    run_roots: list[Path],
    board_paths: list[Path],
    top_k: int,
    include_default_boards: bool = True,
) -> str:
    evidence = load_route_metadata()
    merge_route_summaries(evidence, summary_dir)
    committed_boards = default_protocol_boards() if include_default_boards else []
    merge_protocol_boards(evidence, committed_boards, require_complete_audit=True)
    merge_protocol_boards(evidence, find_protocol_boards(run_roots), require_complete_audit=True)
    merge_protocol_boards(evidence, board_paths, require_complete_audit=True)
    return render_markdown(evidence, top_k=top_k)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate lightweight evaluation from existing summaries and protocol boards.")
    parser.add_argument("--summary-dir", type=Path, default=ROOT / "reports" / "route_summaries")
    parser.add_argument("--run-root", type=Path, action="append", default=[])
    parser.add_argument("--board", type=Path, action="append", default=[])
    parser.add_argument("--no-default-boards", action="store_true", help="Ignore committed compact board snapshots and use only summaries/explicit inputs.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    text = build_report(
        summary_dir=args.summary_dir,
        run_roots=args.run_root,
        board_paths=args.board,
        top_k=args.top_k,
        include_default_boards=not args.no_default_boards,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
