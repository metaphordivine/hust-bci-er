"""Summarize DEP/HC diagnostic runs into a baseline board.

This script is report-only. It reads existing DEP/HC diagnostic outputs and
does not change route status or create candidate evidence.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
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
    "eval_scope",
    "split_id",
    "seed",
    "fold",
    "holdout_seed",
    "feature_or_fusion",
    "classifier",
    "threshold_objective",
    "threshold_source",
    "fusion_weight_source",
    "fusion_weight_by_feature",
    "subject_aggregation",
    "primary_ba",
    "primary_hc_recall",
    "primary_dep_recall",
    "primary_min_recall",
    "primary_recall_gap",
    "primary_dep_vs_hc_ratio",
    "primary_metric_source",
    "crop_combo_status",
    "crop_combo_expected_ba",
    "crop_combo_worst_ba",
    "crop_combo_expected_hc_recall",
    "crop_combo_expected_dep_recall",
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
    "error_type",
    "n_errors",
    "false_negative_errors",
    "false_positive_errors",
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
    paths: list[Path] = []
    matches = [Path(match) for match in glob.glob(str(item))] if item.is_absolute() else sorted(ROOT.glob(str(item)))
    for match in matches:
        if match.is_file() and match.name == "dep_hc_task_diagnostic.json":
            paths.append(match)
        elif match.is_dir():
            direct = match / "dep_hc_task_diagnostic.json"
            if direct.exists():
                paths.append(direct)
    return paths


def _read_subject_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def canonical_feature_order(features: list[Any]) -> list[str]:
    preferred = {
        "traditional": 0,
        "time_frequency": 1,
        "connectivity": 2,
        "graph_connectivity": 3,
    }
    return sorted((str(feature) for feature in features), key=lambda feature: (preferred.get(feature, 100), feature))


def board_row(run: dict[str, Any]) -> dict[str, str]:
    root = Path(run["root"])
    payload = run["payload"]
    metrics = payload.get("metrics", {})
    config = payload.get("config", {})
    if "score_fusion_components" in metrics:
        components = [str(component) for component in metrics["score_fusion_components"]]
        feature_or_fusion = "score:" + "+".join(components)
        fusion_weight_by_feature = json.dumps(metrics.get("score_fusion_weights_by_component", {}), sort_keys=True)
        fusion_weight_source = str(metrics.get("fusion_weight_source", ""))
        classifier = "score_fusion"
    elif "fusion_feature_sets" in metrics:
        feature_or_fusion = "+".join(canonical_feature_order(metrics["fusion_feature_sets"]))
        fusion_weight_by_feature = json.dumps(metrics.get("fusion_weight_by_feature", {}), sort_keys=True)
        fusion_weight_source = str(metrics.get("fusion_weight_source", "validation_subjects"))
        classifier = "fusion"
    elif metrics.get("model_name") or config.get("model_name"):
        feature_or_fusion = f"neural:{metrics.get('model_name', config.get('model_name', ''))}"
        fusion_weight_by_feature = ""
        fusion_weight_source = ""
        classifier = "neural"
    else:
        feature_or_fusion = str(metrics.get("feature_set", config.get("feature_set", "")))
        fusion_weight_by_feature = ""
        fusion_weight_source = ""
        classifier = str(metrics.get("classifier", config.get("classifier", "")))
    combo_status = str(metrics.get("crop_combo_status", ""))
    has_combo = combo_status == "computed" and math.isfinite(_float_metric(metrics, "crop_combo_expected_ba"))
    primary_ba = _float_metric(metrics, "crop_combo_expected_ba") if has_combo else _float_metric(metrics, "subject_ba")
    primary_hc = _float_metric(metrics, "crop_combo_expected_hc_recall") if has_combo else _float_metric(metrics, "hc_subject_recall")
    primary_dep = _float_metric(metrics, "crop_combo_expected_dep_recall") if has_combo else _float_metric(metrics, "dep_subject_recall")
    primary_source = "crop_combo_expected" if has_combo else "legacy_subject_aggregate"
    hc = _float_metric(metrics, "hc_subject_recall")
    dep = _float_metric(metrics, "dep_subject_recall")
    if math.isfinite(hc) and math.isfinite(dep):
        ratio = dep / hc if hc != 0.0 else float("inf")
    else:
        ratio = float("nan")
    if math.isfinite(primary_hc) and math.isfinite(primary_dep):
        primary_ratio = primary_dep / primary_hc if primary_hc != 0.0 else float("inf")
        primary_min_recall = min(primary_hc, primary_dep)
        primary_recall_gap = abs(primary_dep - primary_hc)
    else:
        primary_ratio = float("nan")
        primary_min_recall = float("nan")
        primary_recall_gap = float("nan")
    split_id = str(config.get("split_id", ""))
    holdout_seed = _holdout_seed_from_config(config)
    return {
        "run_id": root.name,
        "protocol": str(config.get("protocol", "")),
        "eval_scope": _eval_scope(config),
        "split_id": split_id,
        "seed": str(config.get("seed", "")),
        "fold": str(config.get("fold", "")),
        "holdout_seed": holdout_seed,
        "feature_or_fusion": feature_or_fusion,
        "classifier": classifier,
        "threshold_objective": str(metrics.get("threshold_objective", config.get("threshold_objective", ""))),
        "threshold_source": str(metrics.get("threshold_source", "")),
        "fusion_weight_source": fusion_weight_source,
        "fusion_weight_by_feature": fusion_weight_by_feature,
        "subject_aggregation": str(metrics.get("subject_aggregation", config.get("subject_aggregation", ""))),
        "primary_ba": _format_float(primary_ba),
        "primary_hc_recall": _format_float(primary_hc),
        "primary_dep_recall": _format_float(primary_dep),
        "primary_min_recall": _format_float(primary_min_recall),
        "primary_recall_gap": _format_float(primary_recall_gap),
        "primary_dep_vs_hc_ratio": _format_float(primary_ratio),
        "primary_metric_source": primary_source,
        "crop_combo_status": combo_status,
        "crop_combo_expected_ba": _format_float(_float_metric(metrics, "crop_combo_expected_ba")),
        "crop_combo_worst_ba": _format_float(_float_metric(metrics, "crop_combo_worst_ba")),
        "crop_combo_expected_hc_recall": _format_float(_float_metric(metrics, "crop_combo_expected_hc_recall")),
        "crop_combo_expected_dep_recall": _format_float(_float_metric(metrics, "crop_combo_expected_dep_recall")),
        "subject_ba": _format_float(_float_metric(metrics, "subject_ba")),
        "hc_subject_recall": _format_float(hc),
        "dep_subject_recall": _format_float(dep),
        "dep_vs_hc_ratio": _format_float(ratio),
        "n_eval_subjects": str(metrics.get("n_eval_subjects", "")),
        "output_root": str(root),
    }


def hard_subject_rows(runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    false_negative_counts: Counter[tuple[str, str]] = Counter()
    false_positive_counts: Counter[tuple[str, str]] = Counter()
    run_names: dict[tuple[str, str], list[str]] = defaultdict(list)
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for run in runs:
        run_id = Path(run["root"]).name
        for row in run["subject_rows"]:
            if row.get("cohort") == row.get("predicted_cohort"):
                continue
            key = (str(row.get("subject_id", "")), str(row.get("cohort", "")))
            counts[key] += 1
            if row.get("cohort") == "DEP" and row.get("predicted_cohort") == "HC":
                false_negative_counts[key] += 1
            elif row.get("cohort") == "HC" and row.get("predicted_cohort") == "DEP":
                false_positive_counts[key] += 1
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
                "error_type": _hard_error_type(cohort),
                "n_errors": str(count),
                "false_negative_errors": str(false_negative_counts[(subject_id, cohort)]),
                "false_positive_errors": str(false_positive_counts[(subject_id, cohort)]),
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
        "Primary comparison uses `crop_combo_expected_*` when available; legacy all-crop subject metrics are kept for audit only.",
        "",
        "## Robust Recommendation",
        "",
        "For DEP/HC defaults, this ranking prioritizes P2 recall balance and worst-combo stability before mean BA. It is diagnostic guidance, not route promotion evidence.",
        "",
        "| rank | scope | feature/fusion | threshold/agg | split coverage | n | mean primary BA | mean worst combo BA | mean min recall | mean recall gap |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(robust_recommendation_rows(aggregate_rows(board_rows))[:10], start=1):
        lines.append(
            "| {rank} | {eval_scope} | {feature_or_fusion} | {threshold_objective}/{subject_aggregation} | {split_coverage} | {n} | "
            "{mean_primary_ba} | {mean_worst_combo_ba} | {mean_min_recall} | {mean_recall_gap} |".format(rank=rank, **row)
        )
    lines.extend(
        [
            "",
            "## Aggregate Summary",
            "",
            "| scope | feature/fusion | threshold/agg | weight source | split coverage | n | mean primary BA | min primary BA | mean worst combo BA | mean HC recall | mean DEP recall | mean min recall | mean recall gap |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows(board_rows):
        lines.append(
            "| {eval_scope} | {feature_or_fusion} | {threshold_objective}/{subject_aggregation} | {fusion_weight_source} | "
            "{split_coverage} | {n} | {mean_primary_ba} | {min_primary_ba} | {mean_worst_combo_ba} | "
            "{mean_hc_recall} | {mean_dep_recall} | {mean_min_recall} | {mean_recall_gap} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
        "## Baseline Runs",
        "",
        "| run | scope | split | feature/fusion | threshold | metric source | primary BA | worst combo BA | HC recall | DEP recall | DEP/HC | legacy subject BA |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(board_rows, key=lambda item: (item["eval_scope"], item["feature_or_fusion"], item["run_id"])):
        lines.append(
            "| {run_id} | {eval_scope} | {split} | {feature_or_fusion} | {threshold_objective}/{threshold_source} | "
            "{primary_metric_source} | {primary_ba} | {crop_combo_worst_ba} | {primary_hc_recall} | "
            "{primary_dep_recall} | {primary_dep_vs_hc_ratio} | {subject_ba} |".format(**{**row, "split": _row_split_label(row)})
        )
    lines.extend(
        [
            "",
            "## Repeated Hard Subjects",
            "",
            "| subject | cohort | error type | errors | FN | FP | mean p(DEP) when wrong | runs |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in hard_rows[:20]:
        lines.append(
            "| {subject_id} | {cohort} | {error_type} | {n_errors} | {false_negative_errors} | "
            "{false_positive_errors} | {mean_subject_score_p_dep} | `{runs}` |".format(**row)
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
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in board_rows:
        key = (
            row["eval_scope"],
            row["feature_or_fusion"],
            row["fusion_weight_source"],
            row["threshold_objective"],
            row["subject_aggregation"],
        )
        grouped[key].append(row)
    out: list[dict[str, str]] = []
    for (eval_scope, feature_or_fusion, fusion_weight_source, threshold_objective, subject_aggregation), rows in sorted(grouped.items()):
        bas = [_parse_optional_float(row["primary_ba"]) for row in rows]
        worst = [_parse_optional_float(row["crop_combo_worst_ba"]) for row in rows]
        hc = [_parse_optional_float(row["primary_hc_recall"]) for row in rows]
        dep = [_parse_optional_float(row["primary_dep_recall"]) for row in rows]
        min_recalls = [_parse_optional_float(row["primary_min_recall"]) for row in rows]
        gaps = [_parse_optional_float(row["primary_recall_gap"]) for row in rows]
        bas = [value for value in bas if math.isfinite(value)]
        worst = [value for value in worst if math.isfinite(value)]
        hc = [value for value in hc if math.isfinite(value)]
        dep = [value for value in dep if math.isfinite(value)]
        min_recalls = [value for value in min_recalls if math.isfinite(value)]
        gaps = [value for value in gaps if math.isfinite(value)]
        out.append(
            {
                "eval_scope": eval_scope,
                "feature_or_fusion": feature_or_fusion,
                "fusion_weight_source": fusion_weight_source,
                "threshold_objective": threshold_objective,
                "subject_aggregation": subject_aggregation,
                "split_coverage": _split_coverage(eval_scope, rows),
                "n": str(len(rows)),
                "mean_primary_ba": _format_float(sum(bas) / len(bas) if bas else float("nan")),
                "min_primary_ba": _format_float(min(bas) if bas else float("nan")),
                "mean_worst_combo_ba": _format_float(sum(worst) / len(worst) if worst else float("nan")),
                "mean_hc_recall": _format_float(sum(hc) / len(hc) if hc else float("nan")),
                "mean_dep_recall": _format_float(sum(dep) / len(dep) if dep else float("nan")),
                "mean_min_recall": _format_float(sum(min_recalls) / len(min_recalls) if min_recalls else float("nan")),
                "mean_recall_gap": _format_float(sum(gaps) / len(gaps) if gaps else float("nan")),
            }
        )
    return out


def robust_recommendation_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    p2_rows = [row for row in rows if row["eval_scope"] == "p2"]
    return sorted(
        p2_rows,
        key=lambda row: (
            _has_finite(row["mean_recall_gap"]),
            _sort_float(row["mean_min_recall"]),
            _sort_float(row["mean_worst_combo_ba"]),
            _sort_float(row["mean_primary_ba"]),
            -_sort_float(row["mean_recall_gap"]),
            int(row["n"]),
        ),
        reverse=True,
    )


def _eval_scope(config: dict[str, Any]) -> str:
    protocol = str(config.get("protocol", ""))
    if protocol == "p1":
        return "p1_full"
    if protocol == "p2":
        return "p2"
    return protocol


def _row_split_label(row: dict[str, str]) -> str:
    if row["eval_scope"] == "p1_full":
        return f"seed{row['seed']}_fold{row['fold']}"
    if row["eval_scope"] == "p2":
        return f"holdout{row['holdout_seed']}" if row["holdout_seed"] else row["split_id"]
    return row["split_id"]


def _split_coverage(eval_scope: str, rows: list[dict[str, str]]) -> str:
    if eval_scope == "p1_full":
        folds = sorted({row["fold"] for row in rows if row["fold"] != ""}, key=lambda value: int(value))
        seeds = sorted({row["seed"] for row in rows if row["seed"] != ""}, key=lambda value: int(value))
        complete = "complete" if {"0", "1", "2", "3", "4"}.issubset(set(folds)) else "partial"
        return f"{complete}; seeds={','.join(seeds) or '-'}; folds={','.join(folds) or '-'}"
    if eval_scope == "p2":
        holdouts = sorted({row["holdout_seed"] for row in rows if row["holdout_seed"] not in {"", "0"}}, key=lambda value: int(value))
        return f"holdouts={','.join(holdouts) or '-'}"
    return "-"


def _holdout_seed_from_config(config: dict[str, Any]) -> str:
    raw = config.get("holdout_seed", "")
    if raw not in {"", None}:
        return str(raw)
    match = re.search(r"(?:^|_)(?:h|holdout)(\d+)(?:$|_)", str(config.get("split_id", "")))
    return match.group(1) if match else ""


def _float_metric(metrics: dict[str, Any], key: str) -> float:
    return _parse_optional_float(metrics.get(key, ""))


def _parse_optional_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_float(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.4f}" if math.isfinite(value) else ""


def _sort_float(value: str) -> float:
    parsed = _parse_optional_float(value)
    return parsed if math.isfinite(parsed) else -1.0


def _has_finite(value: str) -> int:
    return int(math.isfinite(_parse_optional_float(value)))


def _hard_error_type(cohort: str) -> str:
    if cohort == "DEP":
        return "false_negative"
    if cohort == "HC":
        return "false_positive"
    return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
