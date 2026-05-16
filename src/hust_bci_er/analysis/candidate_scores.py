"""Candidate score-matrix diagnostics for model deep dives."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from hust_bci_er.contracts.prediction import canonical_prediction_column, prediction_schema
from hust_bci_er.evaluation.exact_single_crop import top4_predictions
from hust_bci_er.evaluation.metrics import balanced_accuracy
from hust_bci_er.evaluation.report import crop_score_columns, parse_binary, parse_finite_float, score_matrix_metric_rows


DEFAULT_METRIC = "exact_single_crop_expected_BA"


@dataclass(frozen=True)
class ScoreMatrixRun:
    """Loaded score-matrix artifact with the metadata needed for diagnostics."""

    label: str
    route_id: str
    score_matrix_path: Path
    rows: tuple[Mapping[str, str], ...]
    score_columns: tuple[str, ...]
    group_keys: tuple[str, ...]
    subject_column: str
    trial_column: str
    truth_column: str
    manifest_path: Path | None = None
    primary_metric: str = DEFAULT_METRIC


@dataclass(frozen=True)
class RouteDiagnostic:
    label: str
    route_id: str
    score_matrix_path: Path
    exact_ba: float
    mean_score_top4_ba: float
    dep_ba: float | None
    hc_ba: float | None
    n_groups: int
    n_trials: int
    crop_ba: tuple[float, ...]
    calibration: Mapping[str, float | None]


@dataclass(frozen=True)
class CandidateScoreAnalysis:
    runs: tuple[ScoreMatrixRun, ...]
    diagnostics: tuple[RouteDiagnostic, ...]
    correlation_matrix: Mapping[str, Mapping[str, float | None]]
    disagreement_matrix: Mapping[str, Mapping[str, float | None]]
    error_samples: tuple[Mapping[str, Any], ...]


def discover_score_matrices(paths: Sequence[Path]) -> list[Path]:
    """Resolve input files or directories to score_matrix.csv paths."""

    discovered: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            discovered.append(path)
            continue
        direct = path / "score_matrix.csv"
        if direct.exists():
            discovered.append(direct)
            continue
        if path.exists():
            discovered.extend(sorted(path.rglob("score_matrix.csv")))
    unique: dict[Path, Path] = {}
    for path in discovered:
        unique[path.resolve()] = path
    return [unique[key] for key in sorted(unique)]


def load_score_matrix_run(path: Path, *, label: str | None = None) -> ScoreMatrixRun:
    """Load a score_matrix.csv artifact into a :class:`ScoreMatrixRun`.

    The CSV must follow the Top-4 candidate format:

    - Columns: ``subject_id`` (or ``user_id``), ``trial_id``, ``y_true``,
      and exactly five crop score columns (``crop_0`` … ``crop_4``).
    - Each ``subject_id`` group must contain exactly **8 rows** with
      exactly **4 positive** ``y_true`` values.

    Structural constraints are validated at load time. Exact metric values are
    still computed during :func:`analyze_score_matrices`.
    """
    score_matrix_path = Path(path)
    manifest_path = score_matrix_path.parent / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    with score_matrix_path.open(encoding="utf-8", newline="") as f:
        rows = tuple(csv.DictReader(f))
    if not rows:
        raise ValueError(f"score matrix CSV has no rows: {score_matrix_path}")

    fields = set(rows[0])
    schema = prediction_schema(fields)
    subject_col = schema["subject_id"]
    trial_col = schema["trial_id"]
    truth_col = schema["y_true"]
    score_cols = tuple(crop_score_columns(fields))
    if subject_col is None or trial_col is None or truth_col is None or len(score_cols) != 5:
        raise ValueError(f"score matrix must include subject_id, trial_id, y_true, and crop_0..crop_4: {score_matrix_path}")

    route_id = str(manifest.get("route_id") or score_matrix_path.parent.name)
    run_label = label or route_id
    primary_metric = str(manifest.get("primary_metric") or DEFAULT_METRIC)
    group_keys = _metric_group_keys(manifest, fields, schema)

    run = ScoreMatrixRun(
        label=run_label,
        route_id=route_id,
        score_matrix_path=score_matrix_path,
        rows=rows,
        score_columns=score_cols,
        group_keys=group_keys,
        subject_column=subject_col,
        trial_column=trial_col,
        truth_column=truth_col,
        manifest_path=manifest_path if manifest_path.exists() else None,
        primary_metric=primary_metric,
    )
    _validate_score_matrix_groups(run)
    return run


def analyze_score_matrices(paths: Sequence[Path], *, top_errors: int = 40) -> CandidateScoreAnalysis:
    score_paths = discover_score_matrices(paths)
    if not score_paths:
        raise ValueError("no score_matrix.csv files found")
    runs = _dedupe_labels(tuple(load_score_matrix_run(path) for path in score_paths))
    diagnostics = tuple(_diagnose_run(run) for run in runs)
    correlations = _pairwise_score_correlations(runs)
    disagreements = _pairwise_top4_disagreement(runs)
    error_samples = tuple(_top_error_samples(runs, limit=top_errors))
    return CandidateScoreAnalysis(
        runs=runs,
        diagnostics=diagnostics,
        correlation_matrix=correlations,
        disagreement_matrix=disagreements,
        error_samples=error_samples,
    )


def render_markdown_report(analysis: CandidateScoreAnalysis) -> str:
    lines: list[str] = [
        "# Candidate Score Deep Dive",
        "",
        "This report is diagnostic only. It does not change route status and it does not replace candidate or promoted gates.",
        "",
        "## Inputs",
        "",
        "| route | score_matrix | manifest |",
        "|---|---|---|",
    ]
    for run in analysis.runs:
        manifest = _display_path(run.manifest_path) if run.manifest_path is not None else ""
        lines.append(f"| `{run.label}` | `{_display_path(run.score_matrix_path)}` | `{manifest}` |")

    lines.extend(
        [
            "",
            "## Leaderboard",
            "",
            "| route | exact_single_crop_expected_BA | mean_score_top4_BA | DEP exact BA | HC exact BA | groups | trials |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in sorted(analysis.diagnostics, key=lambda diagnostic: diagnostic.exact_ba, reverse=True):
        lines.append(
            "| `{route}` | {exact} | {mean_ba} | {dep} | {hc} | {groups} | {trials} |".format(
                route=item.label,
                exact=_fmt(item.exact_ba),
                mean_ba=_fmt(item.mean_score_top4_ba),
                dep=_fmt_optional(item.dep_ba),
                hc=_fmt_optional(item.hc_ba),
                groups=item.n_groups,
                trials=item.n_trials,
            )
        )

    lines.extend(
        [
            "",
            "## Crop Performance",
            "",
            "| route | crop_0 | crop_1 | crop_2 | crop_3 | crop_4 | best | worst |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in sorted(analysis.diagnostics, key=lambda diagnostic: diagnostic.exact_ba, reverse=True):
        crop_values = list(item.crop_ba)
        best_idx = int(np.nanargmax(crop_values)) if crop_values else -1
        worst_idx = int(np.nanargmin(crop_values)) if crop_values else -1
        lines.append(
            "| `{route}` | {crops} | crop_{best} | crop_{worst} |".format(
                route=item.label,
                crops=" | ".join(_fmt(value) for value in crop_values),
                best=best_idx,
                worst=worst_idx,
            )
        )

    lines.extend(
        [
            "",
            "## Calibration",
            "",
            "| route | mean | std | min | max | abs_from_0.5 | overconfident_frac | pos_mean | neg_mean | brier | outside_0_1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in sorted(analysis.diagnostics, key=lambda diagnostic: diagnostic.exact_ba, reverse=True):
        cal = item.calibration
        lines.append(
            "| `{route}` | {mean} | {std} | {minv} | {maxv} | {abs_mid} | {overconf} | {pos} | {neg} | {brier} | {outside} |".format(
                route=item.label,
                mean=_fmt_optional(cal["mean"]),
                std=_fmt_optional(cal["std"]),
                minv=_fmt_optional(cal["min"]),
                maxv=_fmt_optional(cal["max"]),
                abs_mid=_fmt_optional(cal["mean_abs_from_0_5"]),
                overconf=_fmt_optional(cal["overconfident_fraction"]),
                pos=_fmt_optional(cal["positive_mean"]),
                neg=_fmt_optional(cal["negative_mean"]),
                brier=_fmt_optional(cal["brier"]),
                outside=_fmt_optional(cal["outside_0_1_fraction"]),
            )
        )

    lines.extend(["", "## Score Correlation", ""])
    lines.extend(_render_matrix(analysis.correlation_matrix))
    lines.extend(["", "## Mean-score Top-4 Disagreement", ""])
    lines.extend(_render_matrix(analysis.disagreement_matrix))

    lines.extend(
        [
            "",
            "## Frequent Error Samples",
            "",
            "| subject | trial | y_true | wrong_models | total_models | models |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    if analysis.error_samples:
        for row in analysis.error_samples:
            lines.append(
                "| `{subject}` | `{trial}` | {truth} | {wrong} | {total} | {models} |".format(
                    subject=row["subject_id"],
                    trial=row["trial_id"],
                    truth=row["y_true"],
                    wrong=row["wrong_models"],
                    total=row["total_models"],
                    models=", ".join(f"`{model}`" for model in row["models"]),
                )
            )
    else:
        lines.append("|  |  |  |  |  |  |")

    lines.append("")
    return "\n".join(lines)


def write_markdown_report(analysis: CandidateScoreAnalysis, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown_report(analysis), encoding="utf-8")
    return out_path


def _diagnose_run(run: ScoreMatrixRun) -> RouteDiagnostic:
    subject_rows = score_matrix_metric_rows(
        list(run.rows),
        metric_name=DEFAULT_METRIC,
        metric_group_keys=run.group_keys,
    )
    exact_ba = float(np.mean([float(row["metric_value"]) for row in subject_rows]))
    exact_by_cohort: dict[str, list[float]] = defaultdict(list)
    for row in subject_rows:
        cohort = _cohort(str(row["subject_id"]))
        if cohort is not None:
            exact_by_cohort[cohort].append(float(row["metric_value"]))

    groups = _group_rows(run)
    mean_ba_rows: list[tuple[str, float]] = []
    crop_values: list[list[float]] = [[] for _ in run.score_columns]
    n_trials = 0

    for _, rows in sorted(groups.items()):
        matrix, y_true = _score_matrix_and_truth(run, rows)
        n_trials += len(rows)
        mean_scores = matrix.mean(axis=1)
        mean_pred = top4_predictions(mean_scores)
        mean_ba = balanced_accuracy(y_true, mean_pred)
        subject_id = str(rows[0][run.subject_column])
        mean_ba_rows.append((subject_id, mean_ba))
        for idx in range(matrix.shape[1]):
            crop_pred = top4_predictions(matrix[:, idx])
            crop_values[idx].append(balanced_accuracy(y_true, crop_pred))

    return RouteDiagnostic(
        label=run.label,
        route_id=run.route_id,
        score_matrix_path=run.score_matrix_path,
        exact_ba=exact_ba,
        mean_score_top4_ba=float(np.mean([value for _, value in mean_ba_rows])),
        dep_ba=_mean_or_none(exact_by_cohort["DEP"]),
        hc_ba=_mean_or_none(exact_by_cohort["HC"]),
        n_groups=len(groups),
        n_trials=n_trials,
        crop_ba=tuple(float(np.mean(values)) for values in crop_values),
        calibration=_calibration(run),
    )


def _calibration(run: ScoreMatrixRun) -> Mapping[str, float | None]:
    scores: list[float] = []
    truths: list[int] = []
    for row in run.rows:
        row_scores = [parse_finite_float(row[col], field=col, context=run.label) for col in run.score_columns]
        scores.append(float(np.mean(row_scores)))
        truths.append(parse_binary(row[run.truth_column], field=run.truth_column, context=run.label))
    score_arr = np.asarray(scores, dtype=np.float64)
    truth_arr = np.asarray(truths, dtype=np.int8)
    inside_probability = bool(np.all((score_arr >= 0.0) & (score_arr <= 1.0)))
    positives = score_arr[truth_arr == 1]
    negatives = score_arr[truth_arr == 0]
    return {
        "mean": float(score_arr.mean()),
        "std": float(score_arr.std()),
        "min": float(score_arr.min()),
        "max": float(score_arr.max()),
        "mean_abs_from_0_5": float(np.mean(np.abs(score_arr - 0.5))),
        "overconfident_fraction": float(np.mean((score_arr <= 0.05) | (score_arr >= 0.95))) if inside_probability else None,
        "positive_mean": float(positives.mean()) if positives.size else None,
        "negative_mean": float(negatives.mean()) if negatives.size else None,
        "brier": float(np.mean((score_arr - truth_arr) ** 2)) if inside_probability else None,
        "outside_0_1_fraction": float(np.mean((score_arr < 0.0) | (score_arr > 1.0))),
    }


def _pairwise_score_correlations(runs: Sequence[ScoreMatrixRun]) -> Mapping[str, Mapping[str, float | None]]:
    score_maps = {run.label: _mean_score_map(run) for run in runs}
    return _pairwise_matrix(score_maps, _pearson_for_common_keys, diagonal=1.0)


def _pairwise_top4_disagreement(runs: Sequence[ScoreMatrixRun]) -> Mapping[str, Mapping[str, float | None]]:
    pred_maps = {run.label: _top4_prediction_map(run) for run in runs}
    return _pairwise_matrix(pred_maps, _mismatch_rate_for_common_keys, diagonal=0.0)


def _top_error_samples(runs: Sequence[ScoreMatrixRun], *, limit: int) -> list[Mapping[str, Any]]:
    aggregate: dict[tuple[str, str, str], dict[str, Any]] = {}
    for run in runs:
        pred_map = _top4_prediction_map(run)
        truth_map = _truth_map(run)
        subject_trial = _subject_trial_map(run)
        for key, pred in pred_map.items():
            truth = truth_map[key]
            subject_id, trial_id = subject_trial[key]
            sample_key = (subject_id, trial_id, str(truth))
            item = aggregate.setdefault(
                sample_key,
                {
                    "subject_id": subject_id,
                    "trial_id": trial_id,
                    "y_true": int(truth),
                    "models": [],
                    "total_models": 0,
                    "wrong_models": 0,
                },
            )
            item["total_models"] += 1
            if int(pred) != int(truth):
                item["wrong_models"] += 1
                item["models"].append(run.label)
    rows = [item for item in aggregate.values() if int(item["wrong_models"]) > 0]
    rows.sort(key=lambda item: (-int(item["wrong_models"]), -int(item["total_models"]), str(item["subject_id"]), str(item["trial_id"])))
    return rows[:limit]


def _pairwise_matrix(
    maps: Mapping[str, Mapping[tuple[str, ...], float | int]],
    fn: Any,
    *,
    diagonal: float,
) -> dict[str, dict[str, float | None]]:
    labels = list(maps)
    matrix: dict[str, dict[str, float | None]] = {}
    for left in labels:
        matrix[left] = {}
        for right in labels:
            if left == right:
                matrix[left][right] = diagonal if maps[left] else None
            else:
                matrix[left][right] = fn(maps[left], maps[right])
    return matrix


def _pearson_for_common_keys(
    left: Mapping[tuple[str, ...], float | int],
    right: Mapping[tuple[str, ...], float | int],
) -> float | None:
    keys = sorted(set(left) & set(right))
    if len(keys) < 2:
        return None
    a = np.asarray([left[key] for key in keys], dtype=np.float64)
    b = np.asarray([right[key] for key in keys], dtype=np.float64)
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _mismatch_rate_for_common_keys(
    left: Mapping[tuple[str, ...], float | int],
    right: Mapping[tuple[str, ...], float | int],
) -> float | None:
    keys = sorted(set(left) & set(right))
    if not keys:
        return None
    return float(np.mean([int(left[key]) != int(right[key]) for key in keys]))


def _mean_score_map(run: ScoreMatrixRun) -> dict[tuple[str, ...], float]:
    scores: dict[tuple[str, ...], float] = {}
    for row in run.rows:
        key = _alignment_key(run, row)
        scores[key] = float(np.mean([parse_finite_float(row[col], field=col, context=run.label) for col in run.score_columns]))
    return scores


def _truth_map(run: ScoreMatrixRun) -> dict[tuple[str, ...], int]:
    return {
        _alignment_key(run, row): parse_binary(row[run.truth_column], field=run.truth_column, context=run.label)
        for row in run.rows
    }


def _subject_trial_map(run: ScoreMatrixRun) -> dict[tuple[str, ...], tuple[str, str]]:
    return {
        _alignment_key(run, row): (str(row[run.subject_column]), str(row[run.trial_column]))
        for row in run.rows
    }


def _top4_prediction_map(run: ScoreMatrixRun) -> dict[tuple[str, ...], int]:
    predictions: dict[tuple[str, ...], int] = {}
    for _, rows in _group_rows(run).items():
        mean_scores = np.asarray(
            [
                np.mean([parse_finite_float(row[col], field=col, context=run.label) for col in run.score_columns])
                for row in rows
            ],
            dtype=np.float64,
        )
        pred = top4_predictions(mean_scores)
        for row, value in zip(rows, pred):
            predictions[_alignment_key(run, row)] = int(value)
    return predictions


def _score_matrix_and_truth(run: ScoreMatrixRun, rows: Sequence[Mapping[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [[parse_finite_float(row[col], field=col, context=run.label) for col in run.score_columns] for row in rows],
        dtype=np.float64,
    )
    y_true = np.asarray(
        [parse_binary(row[run.truth_column], field=run.truth_column, context=run.label) for row in rows],
        dtype=np.int8,
    )
    if matrix.shape != (8, 5):
        context = "|".join(str(rows[0][key]) for key in run.group_keys)
        raise ValueError(
            f"Top-4 format requires exactly 8 trials and 5 crop columns per group, "
            f"got shape {matrix.shape}: {run.label} group={context}. "
            "This diagnostic tool only supports the Top-4 candidate score-matrix format."
        )
    trial_ids = [str(row[run.trial_column]) for row in rows]
    if len(set(trial_ids)) != len(trial_ids):
        context = "|".join(str(rows[0][key]) for key in run.group_keys)
        raise ValueError(f"Top-4 format requires unique trial_id values per group: {run.label} group={context}")
    if int(y_true.sum()) != 4:
        context = "|".join(str(rows[0][key]) for key in run.group_keys)
        raise ValueError(
            f"Top-4 format requires exactly 4 positive y_true values per group, "
            f"got {int(y_true.sum())}: {run.label} group={context}"
        )
    return matrix, y_true


def _validate_score_matrix_groups(run: ScoreMatrixRun) -> None:
    for _, rows in sorted(_group_rows(run).items()):
        _score_matrix_and_truth(run, rows)


def _group_rows(run: ScoreMatrixRun) -> dict[tuple[str, ...], list[Mapping[str, str]]]:
    groups: dict[tuple[str, ...], list[Mapping[str, str]]] = defaultdict(list)
    for row in run.rows:
        groups[tuple(str(row[key]) for key in run.group_keys)].append(row)
    return groups


def _alignment_key(run: ScoreMatrixRun, row: Mapping[str, str]) -> tuple[str, ...]:
    key_fields = [key for key in run.group_keys if key != run.trial_column]
    return tuple(str(row[key]) for key in key_fields) + (str(row[run.trial_column]),)


def _metric_group_keys(manifest: Mapping[str, Any], fields: set[str], schema: Mapping[str, str | None]) -> tuple[str, ...]:
    raw = manifest.get("metric_group_keys") or manifest.get("top4_group_keys")
    if isinstance(raw, list) and raw:
        keys: list[str] = []
        missing: list[str] = []
        for item in raw:
            key = str(item)
            column = canonical_prediction_column(key, dict(schema)) or (key if key in fields else None)
            if column is None:
                missing.append(key)
            else:
                keys.append(column)
        if missing:
            raise ValueError(f"score matrix is missing manifest group key(s): {missing}")
        return tuple(keys)
    keys = [key for key in ("seed", "fold") if key in fields]
    subject_key = schema.get("subject_id")
    if subject_key is None:
        raise ValueError("score matrix is missing subject_id group key")
    keys.append(subject_key)
    return tuple(keys)


def _dedupe_labels(runs: Sequence[ScoreMatrixRun]) -> tuple[ScoreMatrixRun, ...]:
    counts: dict[str, int] = defaultdict(int)
    out: list[ScoreMatrixRun] = []
    for run in runs:
        counts[run.label] += 1
        if counts[run.label] == 1:
            out.append(run)
            continue
        suffix = run.score_matrix_path.parent.name
        label = f"{run.label}@{suffix}"
        while any(existing.label == label for existing in out):
            counts[run.label] += 1
            label = f"{run.label}@{counts[run.label]}"
        out.append(
            ScoreMatrixRun(
                label=label,
                route_id=run.route_id,
                score_matrix_path=run.score_matrix_path,
                rows=run.rows,
                score_columns=run.score_columns,
                group_keys=run.group_keys,
                subject_column=run.subject_column,
                trial_column=run.trial_column,
                truth_column=run.truth_column,
                manifest_path=run.manifest_path,
                primary_metric=run.primary_metric,
            )
        )
    return tuple(out)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid manifest JSON: {path}") from exc


def _cohort(subject_id: str) -> str | None:
    upper = subject_id.upper()
    if upper.startswith("DEP"):
        return "DEP"
    if upper.startswith("HC"):
        return "HC"
    return None


def _mean_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def _display_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _fmt(value: float) -> str:
    return f"{value:.6f}" if np.isfinite(value) else ""


def _fmt_optional(value: float | int | None) -> str:
    if value is None:
        return ""
    return _fmt(float(value))


def _render_matrix(matrix: Mapping[str, Mapping[str, float | None]]) -> list[str]:
    labels = list(matrix)
    lines = ["| route | " + " | ".join(f"`{label}`" for label in labels) + " |"]
    lines.append("|---|" + "|".join("---:" for _ in labels) + "|")
    for row_label in labels:
        values = " | ".join(_fmt_optional(matrix[row_label][col_label]) for col_label in labels)
        lines.append(f"| `{row_label}` | {values} |")
    return lines
