"""Metric report builders for prediction and score artifacts."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hust_bci_er.audit.manifest import sha256_file
from hust_bci_er.contracts.prediction import canonical_prediction_column, prediction_schema
from hust_bci_er.evaluation.exact_single_crop import exact_all_correct_rate_from_matrix, exact_ba_from_matrix
from hust_bci_er.evaluation.metrics import all_correct_rate, balanced_accuracy


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prediction_column_for_metric(primary_metric: str, fields: set[str]) -> str:
    schema = prediction_schema(fields)
    if primary_metric == "no_top4_BA":
        column = canonical_prediction_column("y_pred", schema)
        if column is None:
            raise ValueError("no_top4_BA requires y_pred or pred")
        return column
    if primary_metric == "top4_BA":
        column = canonical_prediction_column("pred_top4", schema)
        if column is None:
            raise ValueError("top4_BA requires pred_top4 or y_pred_top4")
        return column
    column = canonical_prediction_column("pred_top4", schema) or canonical_prediction_column("y_pred", schema)
    if column is not None:
        return column
    raise ValueError(f"{primary_metric} requires pred_top4/y_pred_top4 or y_pred/pred")


def subject_ba_rows(
    prediction_rows: list[Mapping[str, Any]],
    *,
    primary_metric: str = "top4_BA",
    metric_group_keys: tuple[str, ...] = ("subject_id",),
) -> list[dict[str, Any]]:
    if not prediction_rows:
        return []
    fields = set(prediction_rows[0])
    schema = prediction_schema(fields)
    truth_col = schema["y_true"]
    if truth_col is None:
        raise ValueError("prediction CSV must include y_true")
    pred_col = prediction_column_for_metric(primary_metric, fields)
    keys = group_columns(fields, schema, metric_group_keys)
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    out = []
    for group_key, rows in sorted(groups.items()):
        context = "|".join(group_key)
        y_true = np.array([parse_binary(row[truth_col], field=truth_col, context=context) for row in rows], dtype=int)
        y_pred = np.array([parse_binary(row[pred_col], field=pred_col, context=context) for row in rows], dtype=int)
        value = balanced_accuracy(y_true, y_pred)
        subject_col = schema["subject_id"]
        subject_id = str(rows[0][subject_col]) if subject_col is not None else group_key[0]
        out.append({"subject_id": subject_id, "group_key": context, "balanced_accuracy": value, "metric_value": value, "n_rows": len(rows)})
    return out


def prediction_metric_rows(
    prediction_rows: list[Mapping[str, Any]],
    *,
    primary_metric: str = "top4_BA",
    metric_group_keys: tuple[str, ...] = ("subject_id",),
) -> list[dict[str, Any]]:
    if primary_metric != "all_correct_rate":
        return subject_ba_rows(prediction_rows, primary_metric=primary_metric, metric_group_keys=metric_group_keys)
    if not prediction_rows:
        return []
    fields = set(prediction_rows[0])
    schema = prediction_schema(fields)
    truth_col = schema["y_true"]
    pred_col = canonical_prediction_column("y_pred", schema)
    if truth_col is None or pred_col is None:
        raise ValueError("all_correct_rate requires y_true and y_pred")
    keys = group_columns(fields, schema, metric_group_keys)
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    out = []
    for group_key, rows in sorted(groups.items()):
        context = "|".join(group_key)
        if len(rows) != 8:
            raise ValueError(f"all-correct group must contain 8 trials: {context}")
        y_true = np.array([[parse_binary(row[truth_col], field=truth_col, context=context) for row in rows]], dtype=int)
        y_pred = np.array([[parse_binary(row[pred_col], field=pred_col, context=context) for row in rows]], dtype=int)
        value = all_correct_rate(y_true, y_pred)
        subject_col = schema["subject_id"]
        subject_id = str(rows[0][subject_col]) if subject_col is not None else group_key[0]
        out.append({"subject_id": subject_id, "group_key": context, "balanced_accuracy": value, "metric_value": value, "n_rows": len(rows)})
    return out


def crop_score_columns(fields: set[str]) -> list[str]:
    for pattern, template in (
        (re.compile(r"^crop_(\d+)$"), "crop_{}"),
        (re.compile(r"^crop(\d+)$"), "crop{}"),
    ):
        indices = sorted(
            int(match.group(1))
            for field in fields
            for match in [pattern.fullmatch(field)]
            if match is not None
        )
        if indices == [0, 1, 2, 3, 4]:
            return [template.format(idx) for idx in range(5)]
    return []


def crop_score_shape_error(fields: set[str]) -> str | None:
    score_like = sorted(field for field in fields if re.fullmatch(r"crop_?\d+", field))
    if crop_score_columns(fields):
        return None
    if score_like:
        return "score matrix metric requires exactly crop_0..crop_4 score columns; found " + ", ".join(score_like)
    return "score matrix CSV must include crop_0..crop_4 score columns"


def group_columns(fields: set[str], schema: dict[str, str | None], raw_keys: tuple[str, ...]) -> list[str]:
    columns: list[str] = []
    for key in raw_keys:
        column = canonical_prediction_column(key, schema) or (key if key in fields else None)
        if column is None:
            raise ValueError(f"score matrix is missing group key: {key}")
        columns.append(column)
    return columns


def parse_binary(value: Any, *, field: str, context: str) -> int:
    parsed = parse_finite_float(value, field=field, context=context)
    if parsed not in {0.0, 1.0}:
        raise ValueError(f"{field} is not binary in {context}: {value}")
    return int(parsed)


def parse_finite_float(value: Any, *, field: str, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric in {context}: {value}") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{field} is not finite in {context}: {value}")
    return parsed


def score_matrix_metric_rows(
    score_matrix_rows: list[Mapping[str, Any]],
    *,
    metric_name: str,
    metric_group_keys: tuple[str, ...] = ("subject_id",),
) -> list[dict[str, Any]]:
    if not score_matrix_rows:
        raise ValueError("score matrix CSV has no rows")
    fields = set(score_matrix_rows[0])
    schema = prediction_schema(fields)
    score_cols = crop_score_columns(fields)
    truth_col = schema["y_true"]
    trial_col = schema["trial_id"]
    shape_error = crop_score_shape_error(fields)
    if shape_error is not None:
        raise ValueError(shape_error)
    if truth_col is None or trial_col is None or len(score_cols) != 5:
        raise ValueError("score matrix CSV must include trial_id, y_true, and crop_0..crop_4 columns")
    keys = group_columns(fields, schema, metric_group_keys)
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in score_matrix_rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)

    out: list[dict[str, Any]] = []
    for group_key, rows in sorted(groups.items()):
        context = "|".join(group_key)
        if len(rows) != 8:
            raise ValueError(f"score matrix group must contain 8 trials: {context}")
        trial_ids = [str(row[trial_col]) for row in rows]
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError(f"score matrix group contains duplicate trial_id values: {context}")
        score_rows = [[parse_finite_float(row[col], field=col, context=context) for col in score_cols] for row in rows]
        labels = [parse_binary(row[truth_col], field=truth_col, context=context) for row in rows]
        if sum(labels) != 4:
            raise ValueError(f"score matrix group must contain exactly 4 positive y_true values: {context}")
        mat = np.array(score_rows, dtype=float)
        y_true = np.array(labels, dtype=int)
        if metric_name == "exact_single_crop_expected_BA":
            value = exact_ba_from_matrix(mat, y_true)
        elif metric_name == "all_correct_rate":
            value = exact_all_correct_rate_from_matrix(mat, y_true)
        else:
            raise ValueError(f"unsupported score matrix metric: {metric_name}")
        subject_col = schema["subject_id"]
        subject_id = str(rows[0][subject_col]) if subject_col is not None else group_key[0]
        out.append({"subject_id": subject_id, "group_key": context, "balanced_accuracy": value, "metric_value": value, "n_rows": len(rows)})
    return out


def build_metric_report(
    *,
    route_id: str,
    prediction_csv: Path | None = None,
    score_matrix_csv: Path | None = None,
    primary_metric: str = "top4_BA",
    metric_group_keys: tuple[str, ...] = ("subject_id",),
) -> dict[str, Any]:
    if score_matrix_csv is not None:
        subject_rows = score_matrix_metric_rows(read_csv_rows(score_matrix_csv), metric_name=primary_metric, metric_group_keys=metric_group_keys)
        mean_value = float(np.mean([row["metric_value"] for row in subject_rows])) if subject_rows else float("nan")
        report = {
            "route_id": route_id,
            "primary_metric": primary_metric,
            "metrics": {primary_metric: mean_value},
            "score_matrix_csv": score_matrix_csv.as_posix(),
            "score_matrix_sha256": sha256_file(score_matrix_csv),
            "subjects": subject_rows,
        }
        if prediction_csv is not None:
            report["prediction_csv"] = prediction_csv.as_posix()
            report["prediction_sha256"] = sha256_file(prediction_csv)
        return report
    if prediction_csv is None:
        raise ValueError("prediction_csv or score_matrix_csv is required")
    rows = read_csv_rows(prediction_csv)
    subject_rows = prediction_metric_rows(rows, primary_metric=primary_metric, metric_group_keys=metric_group_keys)
    mean_value = float(np.mean([row["metric_value"] for row in subject_rows])) if subject_rows else float("nan")
    return {
        "route_id": route_id,
        "primary_metric": primary_metric,
        "metrics": {primary_metric: mean_value},
        "prediction_csv": prediction_csv.as_posix(),
        "prediction_sha256": sha256_file(prediction_csv),
        "subjects": subject_rows,
    }


def write_metric_report(report: Mapping[str, Any], run_dir: Path) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    board = run_dir / "metric_board.csv"
    subjects = run_dir / "subject_ba.csv"
    audit = run_dir / "metric_audit.json"
    with board.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["route_id", "primary_metric", "value"])
        writer.writeheader()
        metric = report["primary_metric"]
        writer.writerow({"route_id": report["route_id"], "primary_metric": metric, "value": report["metrics"][metric]})
    with subjects.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "balanced_accuracy", "metric_value", "n_rows", "group_key"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report.get("subjects", []))
    audit.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"board": board.as_posix(), "subject_ba": subjects.as_posix(), "audit_json": audit.as_posix()}
