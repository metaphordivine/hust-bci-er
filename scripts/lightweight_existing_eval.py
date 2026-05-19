from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
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


def merge_protocol_boards(evidence: dict[str, RouteEvidence], board_paths: list[Path]) -> None:
    for path in board_paths:
        if not path.exists():
            continue
        protocol = _protocol_from_board(path)
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                route_id = str(row.get("route_id") or "").strip()
                if not route_id:
                    continue
                item = evidence.setdefault(route_id, RouteEvidence(route_id=route_id))
                mean = _parse_float(row.get("mean"))
                if mean is not None:
                    item.protocol_means[protocol] = mean
                    item.protocol_sources[protocol] = "protocol_board"
                try:
                    item.protocol_jobs[protocol] = int(float(row.get("n_jobs") or 0))
                except ValueError:
                    pass


def find_protocol_boards(run_roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in run_roots:
        if root.exists():
            paths.extend(sorted(root.rglob("protocol*_board.csv")))
    return paths


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
        "This report is generated from committed route summaries and existing protocol summary CSVs. It does not read raw EEG data, train models, or promote routes.",
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
) -> str:
    evidence = load_route_metadata()
    merge_route_summaries(evidence, summary_dir)
    merge_protocol_boards(evidence, board_paths + find_protocol_boards(run_roots))
    return render_markdown(evidence, top_k=top_k)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate lightweight evaluation from existing summaries and protocol boards.")
    parser.add_argument("--summary-dir", type=Path, default=ROOT / "reports" / "route_summaries")
    parser.add_argument("--run-root", type=Path, action="append", default=[])
    parser.add_argument("--board", type=Path, action="append", default=[])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    text = build_report(summary_dir=args.summary_dir, run_roots=args.run_root, board_paths=args.board, top_k=args.top_k)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
