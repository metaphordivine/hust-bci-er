"""Summarize DEP/HC diagnostic runs into a baseline board.

This script is report-only. It reads existing DEP/HC diagnostic outputs and
does not change route status or create candidate evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


BOARD_FIELDS = [
    "run_id",
    "protocol",
    "split_id",
    "feature_or_fusion",
    "classifier",
    "threshold_objective",
    "threshold_source",
    "fusion_weight_source",
    "fusion_weight_by_feature",
    "subject_aggregation",
    "subject_ba",
    "hc_subject_recall",
    "dep_subject_recall",
    "dep_vs_hc_ratio",
    "n_eval_subjects",
    "output_root",
]

HARD_FIELDS = [
    "subject_id",
    "cohort",
    "n_errors",
    "runs",
    "mean_subject_score_p_dep",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a DEP/HC diagnostic baseline board.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Run directories or dep_hc_task_diagnostic.json files.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "model_deep_dive")
    parser.add_argument("--name", default="dep_hc_baseline_board_20260523")
    args = parser.parse_args(argv)

    runs = load_dep_hc_runs(args.inputs)
    if not runs:
        parser.error("no DEP/HC diagnostic runs found")

    board_rows = [board_row(run) for run in runs]
    hard_rows = hard_subject_rows(runs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    board_csv = args.out_dir / f"{args.name}.csv"
    hard_csv = args.out_dir / f"{args.name}_hard_subjects.csv"
    board_md = args.out_dir / f"{args.name}.md"
    write_csv(board_csv, BOARD_FIELDS, board_rows)
    write_csv(hard_csv, HARD_FIELDS, hard_rows)
    board_md.write_text(render_markdown(board_rows, hard_rows), encoding="utf-8")
    print(json.dumps({"board_csv": str(board_csv), "hard_subject_csv": str(hard_csv), "markdown": str(board_md)}, ensure_ascii=False))
    return 0


def load_dep_hc_runs(inputs: list[Path]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for item in inputs:
        for payload_path in _expand_input(item):
            payload_path = payload_path.resolve()
            if payload_path in seen:
                continue
            seen.add(payload_path)
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            if payload.get("task") != "dep_hc":
                continue
            root = payload_path.parent
            subject_path = root / "dep_hc_subject_metrics.csv"
            subject_rows = _read_subject_rows(subject_path) if subject_path.exists() else []
            runs.append({"root": root, "payload": payload, "subject_rows": subject_rows})
    return sorted(runs, key=lambda run: str(run["root"]))


def _expand_input(item: Path) -> list[Path]:
    if item.is_file():
        return [item] if item.name == "dep_hc_task_diagnostic.json" else []
    if item.is_dir():
        direct = item / "dep_hc_task_diagnostic.json"
        if direct.exists():
            return [direct]
        return sorted(item.glob("*/dep_hc_task_diagnostic.json"))
    return sorted(ROOT.glob(str(item)))


def _read_subject_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def board_row(run: dict[str, Any]) -> dict[str, str]:
    root = Path(run["root"])
    payload = run["payload"]
    metrics = payload.get("metrics", {})
    config = payload.get("config", {})
    if "fusion_feature_sets" in metrics:
        feature_or_fusion = "+".join(str(item) for item in metrics["fusion_feature_sets"])
        fusion_weight_by_feature = json.dumps(metrics.get("fusion_weight_by_feature", {}), sort_keys=True)
        fusion_weight_source = str(metrics.get("fusion_weight_source", "validation_subjects"))
        classifier = "fusion"
    else:
        feature_or_fusion = str(metrics.get("feature_set", config.get("feature_set", "")))
        fusion_weight_by_feature = ""
        fusion_weight_source = ""
        classifier = str(metrics.get("classifier", config.get("classifier", "")))
    hc = _float_metric(metrics, "hc_subject_recall")
    dep = _float_metric(metrics, "dep_subject_recall")
    ratio = dep / hc if hc and math.isfinite(hc) else float("nan")
    return {
        "run_id": root.name,
        "protocol": str(config.get("protocol", "")),
        "split_id": str(config.get("split_id", "")),
        "feature_or_fusion": feature_or_fusion,
        "classifier": classifier,
        "threshold_objective": str(metrics.get("threshold_objective", config.get("threshold_objective", ""))),
        "threshold_source": str(metrics.get("threshold_source", "")),
        "fusion_weight_source": fusion_weight_source,
        "fusion_weight_by_feature": fusion_weight_by_feature,
        "subject_aggregation": str(metrics.get("subject_aggregation", config.get("subject_aggregation", ""))),
        "subject_ba": _format_float(_float_metric(metrics, "subject_ba")),
        "hc_subject_recall": _format_float(hc),
        "dep_subject_recall": _format_float(dep),
        "dep_vs_hc_ratio": _format_float(ratio),
        "n_eval_subjects": str(metrics.get("n_eval_subjects", "")),
        "output_root": str(root),
    }


def hard_subject_rows(runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    run_names: dict[tuple[str, str], list[str]] = defaultdict(list)
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for run in runs:
        run_id = Path(run["root"]).name
        for row in run["subject_rows"]:
            if row.get("cohort") == row.get("predicted_cohort"):
                continue
            key = (str(row.get("subject_id", "")), str(row.get("cohort", "")))
            counts[key] += 1
            run_names[key].append(run_id)
            score = _parse_optional_float(row.get("subject_score_p_dep", row.get("mean_p_dep", "")))
            if math.isfinite(score):
                scores[key].append(score)
    rows: list[dict[str, str]] = []
    for (subject_id, cohort), count in counts.most_common():
        subject_scores = scores[(subject_id, cohort)]
        mean_score = sum(subject_scores) / len(subject_scores) if subject_scores else float("nan")
        rows.append(
            {
                "subject_id": subject_id,
                "cohort": cohort,
                "n_errors": str(count),
                "runs": ";".join(run_names[(subject_id, cohort)]),
                "mean_subject_score_p_dep": _format_float(mean_score),
            }
        )
    return rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(board_rows: list[dict[str, str]], hard_rows: list[dict[str, str]]) -> str:
    lines = [
        "# DEP/HC Diagnostic Baseline Board",
        "",
        "This report is diagnostic-only. It does not change route status and is not candidate evidence.",
        "",
        "## Aggregate Summary",
        "",
        "| protocol | feature/fusion | weight source | n | mean BA | min BA | mean HC recall | mean DEP recall |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows(board_rows):
        lines.append(
            "| {protocol} | {feature_or_fusion} | {fusion_weight_source} | {n} | {mean_subject_ba} | {min_subject_ba} | {mean_hc_recall} | {mean_dep_recall} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
        "## Baseline Runs",
        "",
        "| run | protocol | feature/fusion | threshold | weight source | subject BA | HC recall | DEP recall | DEP/HC |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(board_rows, key=lambda item: (item["protocol"], item["feature_or_fusion"], item["run_id"])):
        lines.append(
            "| {run_id} | {protocol} | {feature_or_fusion} | {threshold_objective}/{threshold_source} | "
            "{fusion_weight_source} | {subject_ba} | {hc_subject_recall} | {dep_subject_recall} | {dep_vs_hc_ratio} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Repeated Hard Subjects",
            "",
            "| subject | cohort | errors | mean p(DEP) when wrong | runs |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in hard_rows[:20]:
        lines.append(
            "| {subject_id} | {cohort} | {n_errors} | {mean_subject_score_p_dep} | `{runs}` |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Use this board to compare diagnostic baselines only.",
            "- Report validation-selected fusion and fixed 0.5/0.5 fusion side by side.",
            "- Subject identifiers shown here are audit/report metadata only, never model features.",
            "- Keep DEP/HC classifier work isolated from emotion Top-4 route evidence until it passes stronger protocol gates.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate_rows(board_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in board_rows:
        key = (row["protocol"], row["feature_or_fusion"], row["fusion_weight_source"])
        grouped[key].append(row)
    out: list[dict[str, str]] = []
    for (protocol, feature_or_fusion, fusion_weight_source), rows in sorted(grouped.items()):
        bas = [_parse_optional_float(row["subject_ba"]) for row in rows]
        hc = [_parse_optional_float(row["hc_subject_recall"]) for row in rows]
        dep = [_parse_optional_float(row["dep_subject_recall"]) for row in rows]
        bas = [value for value in bas if math.isfinite(value)]
        hc = [value for value in hc if math.isfinite(value)]
        dep = [value for value in dep if math.isfinite(value)]
        out.append(
            {
                "protocol": protocol,
                "feature_or_fusion": feature_or_fusion,
                "fusion_weight_source": fusion_weight_source,
                "n": str(len(rows)),
                "mean_subject_ba": _format_float(sum(bas) / len(bas) if bas else float("nan")),
                "min_subject_ba": _format_float(min(bas) if bas else float("nan")),
                "mean_hc_recall": _format_float(sum(hc) / len(hc) if hc else float("nan")),
                "mean_dep_recall": _format_float(sum(dep) / len(dep) if dep else float("nan")),
            }
        )
    return out


def _float_metric(metrics: dict[str, Any], key: str) -> float:
    return _parse_optional_float(metrics.get(key, ""))


def _parse_optional_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_float(value: float) -> str:
    return f"{value:.4f}" if math.isfinite(value) else ""


if __name__ == "__main__":
    raise SystemExit(main())
