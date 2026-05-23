"""Offline gate simulation for DEP/HC subject-type diagnostics.

This script is diagnostic-only. It evaluates whether a DEP/HC subject-type
classifier is reliable enough for future offline expert-routing simulations.
It must not be used to change route status or candidate evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
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
    "subject_ba",
    "hc_recall",
    "dep_recall",
    "n_subjects",
    "n_hc_like",
    "n_dep_like",
    "n_ambiguous",
    "hc_like_precision",
    "dep_like_precision",
    "routable_fraction",
    "passes_gate",
    "failed_reasons",
]

AGG_FIELDS = [
    "protocol",
    "n_runs",
    "mean_subject_ba",
    "min_subject_ba",
    "mean_hc_recall",
    "mean_dep_recall",
    "mean_hc_like_precision",
    "mean_dep_like_precision",
    "mean_routable_fraction",
    "passes_gate",
    "failed_reasons",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate DEP/HC subject-type gate reliability.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Run directories, globs, or dep_hc_task_diagnostic.json files.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "model_deep_dive")
    parser.add_argument("--name", default="dep_hc_subject_type_gate_sim_20260523")
    parser.add_argument("--hc-like-max", type=float, default=0.35)
    parser.add_argument("--dep-like-min", type=float, default=0.65)
    parser.add_argument("--min-mean-ba", type=float, default=0.80)
    parser.add_argument("--min-run-ba", type=float, default=0.625)
    parser.add_argument("--min-recall", type=float, default=0.50)
    parser.add_argument("--min-bucket-precision", type=float, default=0.75)
    args = parser.parse_args(argv)

    if not 0.0 <= args.hc_like_max < args.dep_like_min <= 1.0:
        parser.error("subject-type bounds must satisfy 0 <= hc-like-max < dep-like-min <= 1")

    runs = load_dep_hc_runs(args.inputs)
    if not runs:
        parser.error("no DEP/HC diagnostic runs found")

    run_rows = [
        simulate_run(
            run,
            hc_like_max=args.hc_like_max,
            dep_like_min=args.dep_like_min,
            min_run_ba=args.min_run_ba,
            min_recall=args.min_recall,
            min_bucket_precision=args.min_bucket_precision,
        )
        for run in runs
    ]
    agg_rows = aggregate_gate_rows(
        run_rows,
        min_mean_ba=args.min_mean_ba,
        min_run_ba=args.min_run_ba,
        min_recall=args.min_recall,
        min_bucket_precision=args.min_bucket_precision,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_csv = args.out_dir / f"{args.name}.csv"
    agg_csv = args.out_dir / f"{args.name}_aggregate.csv"
    markdown = args.out_dir / f"{args.name}.md"
    write_csv(run_csv, RUN_FIELDS, run_rows)
    write_csv(agg_csv, AGG_FIELDS, agg_rows)
    markdown.write_text(render_markdown(run_rows, agg_rows, args), encoding="utf-8")
    print(json.dumps({"run_csv": str(run_csv), "aggregate_csv": str(agg_csv), "markdown": str(markdown)}, ensure_ascii=False))
    return 0


def simulate_run(
    run: dict[str, Any],
    *,
    hc_like_max: float,
    dep_like_min: float,
    min_run_ba: float,
    min_recall: float,
    min_bucket_precision: float,
) -> dict[str, str]:
    root = Path(run["root"])
    payload = run["payload"]
    config = payload.get("config", {})
    metrics = payload.get("metrics", {})
    subjects = list(run["subject_rows"])
    buckets = {"hc_like": [], "dep_like": [], "ambiguous": []}
    for row in subjects:
        p_dep = _to_float(row.get("subject_score_p_dep", row.get("mean_p_dep", "")))
        if p_dep <= hc_like_max:
            buckets["hc_like"].append(row)
        elif p_dep >= dep_like_min:
            buckets["dep_like"].append(row)
        else:
            buckets["ambiguous"].append(row)
    subject_ba = _to_float(metrics.get("subject_ba"))
    hc_recall = _to_float(metrics.get("hc_subject_recall"))
    dep_recall = _to_float(metrics.get("dep_subject_recall"))
    hc_precision = _bucket_precision(buckets["hc_like"], expected="HC")
    dep_precision = _bucket_precision(buckets["dep_like"], expected="DEP")
    routable = (len(buckets["hc_like"]) + len(buckets["dep_like"])) / len(subjects) if subjects else float("nan")
    failed = _run_failures(
        subject_ba=subject_ba,
        hc_recall=hc_recall,
        dep_recall=dep_recall,
        hc_precision=hc_precision,
        dep_precision=dep_precision,
        min_run_ba=min_run_ba,
        min_recall=min_recall,
        min_bucket_precision=min_bucket_precision,
    )
    return {
        "run_id": root.name,
        "protocol": str(config.get("protocol", "")),
        "split_id": str(config.get("split_id", "")),
        "subject_ba": _fmt(subject_ba),
        "hc_recall": _fmt(hc_recall),
        "dep_recall": _fmt(dep_recall),
        "n_subjects": str(len(subjects)),
        "n_hc_like": str(len(buckets["hc_like"])),
        "n_dep_like": str(len(buckets["dep_like"])),
        "n_ambiguous": str(len(buckets["ambiguous"])),
        "hc_like_precision": _fmt(hc_precision),
        "dep_like_precision": _fmt(dep_precision),
        "routable_fraction": _fmt(routable),
        "passes_gate": str(not failed).lower(),
        "failed_reasons": ";".join(failed),
    }


def aggregate_gate_rows(
    run_rows: list[dict[str, str]],
    *,
    min_mean_ba: float,
    min_run_ba: float,
    min_recall: float,
    min_bucket_precision: float,
) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in run_rows:
        groups[row["protocol"]].append(row)
    out: list[dict[str, str]] = []
    for protocol, rows in sorted(groups.items()):
        bas = [_to_float(row["subject_ba"]) for row in rows]
        hc = [_to_float(row["hc_recall"]) for row in rows]
        dep = [_to_float(row["dep_recall"]) for row in rows]
        hc_prec = [_to_float(row["hc_like_precision"]) for row in rows if row["hc_like_precision"]]
        dep_prec = [_to_float(row["dep_like_precision"]) for row in rows if row["dep_like_precision"]]
        routable = [_to_float(row["routable_fraction"]) for row in rows]
        failures = []
        if _mean(bas) < min_mean_ba:
            failures.append("mean_ba")
        if min(bas) < min_run_ba:
            failures.append("min_ba")
        if min(hc) < min_recall:
            failures.append("hc_recall")
        if min(dep) < min_recall:
            failures.append("dep_recall")
        if hc_prec and min(hc_prec) < min_bucket_precision:
            failures.append("hc_like_precision")
        if dep_prec and min(dep_prec) < min_bucket_precision:
            failures.append("dep_like_precision")
        if any(row["passes_gate"] != "true" for row in rows):
            failures.append("run_level")
        out.append(
            {
                "protocol": protocol,
                "n_runs": str(len(rows)),
                "mean_subject_ba": _fmt(_mean(bas)),
                "min_subject_ba": _fmt(min(bas)),
                "mean_hc_recall": _fmt(_mean(hc)),
                "mean_dep_recall": _fmt(_mean(dep)),
                "mean_hc_like_precision": _fmt(_mean(hc_prec)),
                "mean_dep_like_precision": _fmt(_mean(dep_prec)),
                "mean_routable_fraction": _fmt(_mean(routable)),
                "passes_gate": str(not failures).lower(),
                "failed_reasons": ";".join(dict.fromkeys(failures)),
            }
        )
    return out


def _run_failures(
    *,
    subject_ba: float,
    hc_recall: float,
    dep_recall: float,
    hc_precision: float,
    dep_precision: float,
    min_run_ba: float,
    min_recall: float,
    min_bucket_precision: float,
) -> list[str]:
    failures: list[str] = []
    if subject_ba < min_run_ba:
        failures.append("subject_ba")
    if hc_recall < min_recall:
        failures.append("hc_recall")
    if dep_recall < min_recall:
        failures.append("dep_recall")
    if math.isfinite(hc_precision) and hc_precision < min_bucket_precision:
        failures.append("hc_like_precision")
    if math.isfinite(dep_precision) and dep_precision < min_bucket_precision:
        failures.append("dep_like_precision")
    return failures


def render_markdown(run_rows: list[dict[str, str]], agg_rows: list[dict[str, str]], args: argparse.Namespace) -> str:
    lines = [
        "# DEP/HC Subject-Type Gate Simulation",
        "",
        "This report is diagnostic-only. It must not be used as candidate evidence or wired into competition routes.",
        "",
        f"Subject-type buckets: HC-like `p_dep <= {args.hc_like_max:.2f}`, DEP-like `p_dep >= {args.dep_like_min:.2f}`, otherwise ambiguous.",
        "",
        "## Aggregate Gate",
        "",
        "| protocol | runs | mean BA | min BA | mean HC recall | mean DEP recall | HC-like precision | DEP-like precision | routable | pass | failed reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in agg_rows:
        lines.append(
            "| {protocol} | {n_runs} | {mean_subject_ba} | {min_subject_ba} | {mean_hc_recall} | {mean_dep_recall} | {mean_hc_like_precision} | {mean_dep_like_precision} | {mean_routable_fraction} | {passes_gate} | {failed_reasons} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Run Gate",
            "",
            "| run | protocol | BA | HC recall | DEP recall | HC-like | DEP-like | ambiguous | HC-like precision | DEP-like precision | pass | failed reasons |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in run_rows:
        lines.append(
            "| {run_id} | {protocol} | {subject_ba} | {hc_recall} | {dep_recall} | {n_hc_like} | {n_dep_like} | {n_ambiguous} | {hc_like_precision} | {dep_like_precision} | {passes_gate} | {failed_reasons} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `passes_gate=false` means the classifier should remain diagnostic-only.",
            "- Expert routing should remain offline-only until both P1 and P2 aggregate gates pass.",
            "- Subject identifiers in this report are audit metadata only and are not model features.",
            "",
        ]
    )
    return "\n".join(lines)


def _bucket_precision(rows: list[dict[str, str]], *, expected: str) -> float:
    if not rows:
        return float("nan")
    return sum(1 for row in rows if row.get("cohort") == expected) / len(rows)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: float) -> str:
    return f"{value:.4f}" if math.isfinite(value) else ""


if __name__ == "__main__":
    raise SystemExit(main())
