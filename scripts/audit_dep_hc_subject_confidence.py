"""Audit subject-level DEP/HC confidence and high-confidence errors.

This is a diagnostic-only helper. It consumes existing DEP/HC subject metrics
and does not change route status or produce candidate evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.summarize_dep_hc_diagnostics import load_dep_hc_runs  # noqa: E402


RUN_FIELDS = [
    "run_id",
    "protocol",
    "split_id",
    "n_subjects",
    "n_errors",
    "n_high_conf_errors",
    "n_ambiguous",
    "max_wrong_confidence",
    "mean_wrong_confidence",
    "output_root",
]

SUBJECT_FIELDS = [
    "run_id",
    "protocol",
    "subject_id",
    "cohort",
    "predicted_cohort",
    "subject_score_p_dep",
    "confidence",
    "bucket",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit DEP/HC subject confidence diagnostics.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Run directories, globs, or dep_hc_task_diagnostic.json files.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "model_deep_dive")
    parser.add_argument("--name", default="dep_hc_subject_confidence_audit_20260523")
    parser.add_argument("--high-confidence", type=float, default=0.75)
    parser.add_argument("--ambiguous-low", type=float, default=0.35)
    parser.add_argument("--ambiguous-high", type=float, default=0.65)
    args = parser.parse_args(argv)

    if args.ambiguous_low >= args.ambiguous_high:
        parser.error("--ambiguous-low must be less than --ambiguous-high")
    if args.high_confidence <= 0.5 or args.high_confidence > 1.0:
        parser.error("--high-confidence must be in (0.5, 1.0]")

    runs = load_dep_hc_runs(args.inputs)
    if not runs:
        parser.error("no DEP/HC diagnostic runs found")

    run_rows: list[dict[str, str]] = []
    subject_rows: list[dict[str, str]] = []
    for run in runs:
        audit = audit_run(
            run,
            high_confidence=args.high_confidence,
            ambiguous_low=args.ambiguous_low,
            ambiguous_high=args.ambiguous_high,
        )
        run_rows.append(audit["run_row"])
        subject_rows.extend(audit["subject_rows"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_csv = args.out_dir / f"{args.name}.csv"
    subject_csv = args.out_dir / f"{args.name}_subjects.csv"
    markdown = args.out_dir / f"{args.name}.md"
    write_csv(run_csv, RUN_FIELDS, run_rows)
    write_csv(subject_csv, SUBJECT_FIELDS, subject_rows)
    markdown.write_text(render_markdown(run_rows, subject_rows, args.high_confidence, args.ambiguous_low, args.ambiguous_high), encoding="utf-8")
    print(json.dumps({"run_csv": str(run_csv), "subject_csv": str(subject_csv), "markdown": str(markdown)}, ensure_ascii=False))
    return 0


def audit_run(
    run: dict[str, Any],
    *,
    high_confidence: float,
    ambiguous_low: float,
    ambiguous_high: float,
) -> dict[str, Any]:
    root = Path(run["root"])
    payload = run["payload"]
    config = payload.get("config", {})
    wrong_conf: list[float] = []
    subject_rows: list[dict[str, str]] = []
    n_ambiguous = 0
    n_errors = 0
    n_high_conf_errors = 0
    for row in run["subject_rows"]:
        p_dep = _to_float(row.get("subject_score_p_dep", row.get("mean_p_dep", "")))
        confidence = _to_float(row.get("confidence", max(p_dep, 1.0 - p_dep)))
        correct = str(row.get("correct", "")) == "1" or row.get("cohort") == row.get("predicted_cohort")
        ambiguous = ambiguous_low <= p_dep <= ambiguous_high
        if ambiguous:
            n_ambiguous += 1
        if correct:
            continue
        n_errors += 1
        wrong_conf.append(confidence)
        high_conf_wrong = confidence >= high_confidence
        if high_conf_wrong:
            n_high_conf_errors += 1
        subject_rows.append(
            {
                "run_id": root.name,
                "protocol": str(config.get("protocol", "")),
                "subject_id": str(row.get("subject_id", "")),
                "cohort": str(row.get("cohort", "")),
                "predicted_cohort": str(row.get("predicted_cohort", "")),
                "subject_score_p_dep": _fmt(p_dep),
                "confidence": _fmt(confidence),
                "bucket": "high_conf_error" if high_conf_wrong else "error",
            }
        )
    run_row = {
        "run_id": root.name,
        "protocol": str(config.get("protocol", "")),
        "split_id": str(config.get("split_id", "")),
        "n_subjects": str(len(run["subject_rows"])),
        "n_errors": str(n_errors),
        "n_high_conf_errors": str(n_high_conf_errors),
        "n_ambiguous": str(n_ambiguous),
        "max_wrong_confidence": _fmt(max(wrong_conf) if wrong_conf else float("nan")),
        "mean_wrong_confidence": _fmt(sum(wrong_conf) / len(wrong_conf) if wrong_conf else float("nan")),
        "output_root": str(root),
    }
    return {"run_row": run_row, "subject_rows": subject_rows}


def render_markdown(
    run_rows: list[dict[str, str]],
    subject_rows: list[dict[str, str]],
    high_confidence: float,
    ambiguous_low: float,
    ambiguous_high: float,
) -> str:
    lines = [
        "# DEP/HC Subject Confidence Audit",
        "",
        "This report is diagnostic-only and must not be used as candidate evidence or route status.",
        "",
        f"High-confidence error threshold: `{high_confidence:.2f}`. Ambiguous p(DEP) interval: `[{ambiguous_low:.2f}, {ambiguous_high:.2f}]`.",
        "",
        "## Run Summary",
        "",
        "| run | protocol | errors | high-conf errors | ambiguous | max wrong conf | mean wrong conf |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in run_rows:
        lines.append(
            "| {run_id} | {protocol} | {n_errors} | {n_high_conf_errors} | {n_ambiguous} | {max_wrong_confidence} | {mean_wrong_confidence} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## High-Confidence Errors",
            "",
            "| run | protocol | subject | cohort | predicted | p(DEP) | confidence |",
            "|---|---|---|---|---|---:|---:|",
        ]
    )
    for row in subject_rows:
        if row["bucket"] != "high_conf_error":
            continue
        lines.append(
            "| {run_id} | {protocol} | {subject_id} | {cohort} | {predicted_cohort} | {subject_score_p_dep} | {confidence} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: float) -> str:
    return f"{value:.4f}" if value == value else ""


if __name__ == "__main__":
    raise SystemExit(main())
