from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from hust_bci_er.audit.manifest import load_manifest, validate_manifest  # noqa: E402
from hust_bci_er.config.registry import AUDIT_DECISIONS  # noqa: E402
from hust_bci_er.config.schema import validate_route_config  # noqa: E402
from hust_bci_er.contracts.prediction import canonical_prediction_column, prediction_schema  # noqa: E402
from hust_bci_er.evaluation.exact_single_crop import exact_all_correct_rate_from_matrix, exact_ba_from_matrix  # noqa: E402
from hust_bci_er.evaluation.metrics import balanced_accuracy  # noqa: E402
from hust_bci_er.inference.topk import topk_binary  # noqa: E402


AuditCheck = dict[str, Any]
GATES = {"smoke", "diagnostic", "candidate", "promoted"}
STRICT_GATES = {"candidate", "promoted"}


def add_check(
    checks: list[AuditCheck],
    *,
    rule_id: str,
    severity: str,
    status: str,
    message: str,
    fix: str = "",
    decision_if_fail: str | None = None,
) -> None:
    checks.append(
        {
            "rule_id": rule_id,
            "severity": severity,
            "status": status,
            "message": message,
            "fix": fix,
            "decision_if_fail": decision_if_fail,
        }
    )


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"route config must be a mapping: {path}")
    return data


def resolve_from_root(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def route_primary_metric(route_data: dict[str, Any]) -> str | None:
    evaluation = route_data.get("evaluation")
    if isinstance(evaluation, dict):
        metric = evaluation.get("primary_metric")
        return str(metric) if metric is not None else None
    return None


def manifest_metric_value(manifest: dict[str, Any], metric_name: str) -> float | None:
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict) or metric_name not in metrics:
        return None
    try:
        return float(metrics[metric_name])
    except (TypeError, ValueError):
        return None


def compare_metric(checks: list[AuditCheck], *, metric_name: str, recomputed: float, manifest: dict[str, Any], tolerance: float = 1e-9) -> None:
    reported = manifest_metric_value(manifest, metric_name)
    if reported is None:
        add_check(
            checks,
            rule_id="PRIMARY_METRIC_REPORTED",
            severity="ERROR",
            status="FAIL",
            message=f"manifest.metrics has no numeric value for {metric_name}",
            fix="Record the reported primary metric in manifest.metrics.",
            decision_if_fail="BLOCKED",
        )
        return
    if abs(recomputed - reported) > tolerance:
        add_check(
            checks,
            rule_id="PRIMARY_METRIC_REPORTED",
            severity="ERROR",
            status="FAIL",
            message=f"reported {metric_name}={reported:.12g} does not match recomputed {recomputed:.12g}",
            fix="Regenerate the metric report from prediction artifacts.",
            decision_if_fail="BLOCKED",
        )
    else:
        add_check(checks, rule_id="PRIMARY_METRIC_REPORTED", severity="INFO", status="PASS", message=f"reported {metric_name} matches recompute")


def route_uses_top4(route_data: dict[str, Any]) -> bool:
    inference = route_data.get("inference")
    return isinstance(inference, dict) and inference.get("top4") is True


def check_split_manifest(route_data: dict[str, Any], checks: list[AuditCheck], *, gate: str) -> None:
    split_id = route_data.get("split_id")
    if not isinstance(split_id, str) or not split_id:
        return

    split_path = ROOT / "configs" / "splits" / f"{split_id}.yaml"
    if not split_path.exists():
        add_check(
            checks,
            rule_id="SPLIT_MANIFEST_EXISTS",
            severity="CRITICAL",
            status="FAIL",
            message=f"split manifest is missing: {split_path.relative_to(ROOT)}",
            fix="Add configs/splits/<split_id>.yaml before auditing this route.",
            decision_if_fail="REJECT",
        )
        return

    split_data = load_yaml(split_path)
    if split_data.get("split_id") != split_id:
        add_check(
            checks,
            rule_id="SPLIT_MANIFEST_ID_MATCH",
            severity="CRITICAL",
            status="FAIL",
            message=f"split manifest id does not match route split_id: {split_id}",
            fix="Make split_id match the split manifest file name and contents.",
            decision_if_fail="REJECT",
        )
        return

    is_placeholder = split_data.get("status") == "declared_without_subject_list"
    has_subject_lists = all(isinstance(split_data.get(key), list) and bool(split_data.get(key)) for key in ["train_subjects", "val_subjects", "test_subjects"])
    folds = split_data.get("folds") or split_data.get("fold_definitions") or []
    has_fold_definitions = isinstance(folds, list) and bool(folds) and all(
        isinstance(fold, dict)
        and all(isinstance(fold.get(key), list) and bool(fold.get(key)) for key in ["train_subjects", "val_subjects", "test_subjects"])
        for fold in folds
    )
    trial_rows = split_data.get("trial_rows") or []
    has_trial_rows = isinstance(trial_rows, list) and bool(trial_rows) and all(
        isinstance(row, dict) and {"subject_id", "original_trial_id", "split"}.issubset(row) for row in trial_rows
    )
    has_verifiable_content = has_subject_lists or has_fold_definitions or has_trial_rows
    if is_placeholder or not has_verifiable_content:
        add_warn_or_fail(
            checks,
            gate=gate,
            fail_gate=STRICT_GATES,
            rule_id="SPLIT_MANIFEST_VERIFIABLE",
            message=f"split manifest is only a placeholder: {split_path.relative_to(ROOT)}",
            fix="Add complete train/val/test subject lists, fold definitions, or trial rows before candidate/promoted review.",
            decision_if_fail="BLOCKED",
        )
    else:
        add_check(checks, rule_id="SPLIT_MANIFEST_VERIFIABLE", severity="INFO", status="PASS", message="split manifest has verifiable split content")


def check_dataset_manifest(route_data: dict[str, Any], checks: list[AuditCheck], *, gate: str) -> None:
    dataset_version = route_data.get("dataset_version")
    if not isinstance(dataset_version, str) or not dataset_version:
        return

    dataset_path = ROOT / "configs" / "datasets" / f"{dataset_version}.yaml"
    if not dataset_path.exists():
        add_check(
            checks,
            rule_id="DATASET_MANIFEST_EXISTS",
            severity="CRITICAL",
            status="FAIL",
            message=f"dataset manifest is missing: {dataset_path.relative_to(ROOT)}",
            fix="Add configs/datasets/<dataset_version>.yaml before auditing this route.",
            decision_if_fail="REJECT",
        )
        return

    dataset_data = load_yaml(dataset_path)
    if dataset_data.get("dataset_version") != dataset_version:
        add_check(
            checks,
            rule_id="DATASET_MANIFEST_ID_MATCH",
            severity="CRITICAL",
            status="FAIL",
            message=f"dataset manifest id does not match route dataset_version: {dataset_version}",
            fix="Make dataset_version match the dataset manifest file name and contents.",
            decision_if_fail="REJECT",
        )
        return

    is_placeholder = dataset_data.get("status") == "declared_without_raw_data_index"
    data_sources = dataset_data.get("data_sources") or []
    checksum_manifest = dataset_data.get("checksum_manifest") or []
    has_data_sources = isinstance(data_sources, list) and bool(data_sources) and all(
        isinstance(item, dict) and item.get("path") and item.get("kind") for item in data_sources
    )
    has_checksums = isinstance(checksum_manifest, list) and bool(checksum_manifest) and all(
        isinstance(item, dict) and item.get("path") and item.get("sha256") for item in checksum_manifest
    )
    has_data_index = has_data_sources and has_checksums
    if is_placeholder or not has_data_index:
        add_warn_or_fail(
            checks,
            gate=gate,
            fail_gate=STRICT_GATES,
            rule_id="DATASET_MANIFEST_VERIFIABLE",
            message=f"dataset manifest is only a placeholder: {dataset_path.relative_to(ROOT)}",
            fix="Add non-empty data_sources and checksum_manifest before candidate/promoted review.",
            decision_if_fail="BLOCKED",
        )
    else:
        add_check(checks, rule_id="DATASET_MANIFEST_VERIFIABLE", severity="INFO", status="PASS", message="dataset manifest has verifiable data evidence")


def add_warn_or_fail(
    checks: list[AuditCheck],
    *,
    gate: str,
    fail_gate: set[str],
    rule_id: str,
    message: str,
    fix: str,
    decision_if_fail: str = "BLOCKED",
) -> None:
    if gate in fail_gate:
        add_check(
            checks,
            rule_id=rule_id,
            severity="ERROR",
            status="FAIL",
            message=message,
            fix=fix,
            decision_if_fail=decision_if_fail,
        )
    else:
        add_check(checks, rule_id=rule_id, severity="WARN", status="WARN", message=message, fix=fix)


def top4_group_columns(fields: set[str], schema: dict[str, str | None], manifest: dict[str, Any]) -> list[str] | None:
    raw_keys = manifest.get("top4_group_keys") or ["subject_id"]
    if not isinstance(raw_keys, list) or not raw_keys:
        return None
    columns: list[str] = []
    for key in raw_keys:
        if not isinstance(key, str):
            return None
        column = canonical_prediction_column(key, schema)
        if column is None:
            column = key if key in fields else None
        if column is None:
            return None
        columns.append(column)
    return columns


def group_prediction_rows(rows: list[dict[str, str]], group_columns: list[str]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        group_id = "|".join(str(row[col]) for col in group_columns)
        groups.setdefault(group_id, []).append(row)
    return groups


def check_top4_group_semantics(
    checks: list[AuditCheck],
    *,
    groups: dict[str, list[dict[str, str]]],
    schema: dict[str, str | None],
) -> None:
    trial_col = schema["trial_id"]
    top4_col = schema["pred_top4"]
    score_col = schema["score"]
    truth_col = schema["y_true"]
    if trial_col is None or top4_col is None:
        return

    bad_unique: list[str] = []
    bad_binary: list[str] = []
    bad_truth: list[str] = []
    bad_ranking: list[str] = []
    for group_id, group in groups.items():
        trial_ids = [str(row[trial_col]) for row in group]
        if len(set(trial_ids)) != len(trial_ids):
            bad_unique.append(group_id)

        observed: list[int] = []
        for row in group:
            try:
                value = float(row[top4_col])
            except ValueError:
                bad_binary.append(group_id)
                value = float("nan")
            if value not in {0.0, 1.0}:
                bad_binary.append(group_id)
            observed.append(int(value) if value in {0.0, 1.0} else -1)

        if truth_col is not None:
            truth_values: list[int] = []
            for row in group:
                try:
                    value = float(row[truth_col])
                except ValueError:
                    bad_truth.append(group_id)
                    value = float("nan")
                if value not in {0.0, 1.0}:
                    bad_truth.append(group_id)
                truth_values.append(int(value) if value in {0.0, 1.0} else -1)
            if len(group) == 8 and sum(truth_values) != 4:
                bad_truth.append(group_id)

        if score_col is not None and len(group) == 8 and all(value in {0, 1} for value in observed):
            scores = np.array([float(row[score_col]) for row in group], dtype=float)
            expected = topk_binary(scores, 4)
            if not np.array_equal(expected, np.array(observed, dtype=int)):
                bad_ranking.append(group_id)

    for rule_id, bad, message, fix in [
        ("PREDICTION_TRIAL_ID_UNIQUE", bad_unique, "duplicate trial_id values inside Top-4 groups", "Ensure each Top-4 group has 8 unique trial_id values."),
        ("PREDICTION_TOP4_BINARY", bad_binary, "pred_top4 contains non-binary values", "Encode pred_top4 as 0 or 1 only."),
        ("PREDICTION_TOP4_TRUTH_BALANCE", bad_truth, "y_true is not binary or not 4 positives per 8-trial group", "For labeled Top-4 audits, each 8-trial group should contain 4 positive labels."),
        ("PREDICTION_TOP4_RANKING", bad_ranking, "pred_top4 does not match score-derived top 4", "Regenerate pred_top4 from score using the repository Top-4 policy."),
    ]:
        if bad:
            add_check(
                checks,
                rule_id=rule_id,
                severity="ERROR",
                status="FAIL",
                message=f"{message}: {', '.join(sorted(set(bad))[:5])}",
                fix=fix,
                decision_if_fail="DIAGNOSTIC_ONLY",
            )
        else:
            add_check(checks, rule_id=rule_id, severity="INFO", status="PASS", message="Top-4 semantic check passed")


def metric_group_columns(fields: set[str], schema: dict[str, str | None], manifest: dict[str, Any]) -> list[str] | None:
    raw_keys = manifest.get("metric_group_keys") or manifest.get("top4_group_keys") or ["subject_id"]
    if not isinstance(raw_keys, list) or not raw_keys:
        return None
    columns: list[str] = []
    for key in raw_keys:
        if not isinstance(key, str):
            return None
        column = canonical_prediction_column(key, schema) or (key if key in fields else None)
        if column is None:
            return None
        columns.append(column)
    return columns


def crop_score_columns(fields: set[str]) -> list[str]:
    candidates = [f"crop_{idx}" for idx in range(5)]
    if all(col in fields for col in candidates):
        return candidates
    candidates = [f"crop{idx}" for idx in range(5)]
    if all(col in fields for col in candidates):
        return candidates
    return []


def recompute_exact_metric_from_matrix(path: Path, *, metric_name: str, manifest: dict[str, Any]) -> float:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    schema = prediction_schema(fields)
    group_cols = metric_group_columns(fields, schema, manifest)
    score_cols = crop_score_columns(fields)
    truth_col = schema["y_true"]
    if group_cols is None or truth_col is None or len(score_cols) != 5:
        raise ValueError("score matrix CSV must include group keys, y_true, and crop_0..crop_4 columns")
    groups = group_prediction_rows(rows, group_cols)
    values: list[float] = []
    for group_id, group in groups.items():
        if len(group) != 8:
            raise ValueError(f"exact metric group must contain 8 trials: {group_id}")
        mat = np.array([[float(row[col]) for col in score_cols] for row in group], dtype=float)
        y_true = np.array([int(float(row[truth_col])) for row in group], dtype=int)
        if metric_name == "exact_single_crop_expected_BA":
            values.append(exact_ba_from_matrix(mat, y_true))
        elif metric_name == "all_correct_rate":
            values.append(exact_all_correct_rate_from_matrix(mat, y_true))
        else:
            raise ValueError(f"unsupported exact metric: {metric_name}")
    if not values:
        raise ValueError("score matrix CSV has no metric groups")
    return float(np.mean(values))


def recompute_primary_metric_from_predictions(
    *,
    metric_name: str,
    rows: list[dict[str, str]],
    fields: set[str],
    schema: dict[str, str | None],
    manifest: dict[str, Any],
) -> float:
    if metric_name in {"no_top4_BA", "top4_BA"}:
        if schema["y_true"] is None or schema["y_pred"] is None:
            raise ValueError("prediction CSV must include y_true and y_pred")
        y_true = np.array([int(float(row[schema["y_true"]])) for row in rows], dtype=int)
        y_pred = np.array([int(float(row[schema["y_pred"]])) for row in rows], dtype=int)
        return balanced_accuracy(y_true, y_pred)

    if metric_name == "all_correct_rate":
        if schema["y_true"] is None or schema["y_pred"] is None:
            raise ValueError("prediction CSV must include y_true and y_pred")
        group_cols = metric_group_columns(fields, schema, manifest)
        if group_cols is None:
            raise ValueError("metric group keys are missing")
        groups = group_prediction_rows(rows, group_cols)
        y_true_matrix = []
        y_pred_matrix = []
        for group_id, group in groups.items():
            if len(group) != 8:
                raise ValueError(f"all-correct group must contain 8 trials: {group_id}")
            y_true_matrix.append([int(float(row[schema["y_true"]])) for row in group])
            y_pred_matrix.append([int(float(row[schema["y_pred"]])) for row in group])
        from hust_bci_er.evaluation.metrics import all_correct_rate

        return all_correct_rate(np.array(y_true_matrix), np.array(y_pred_matrix))

    raise ValueError(f"primary metric requires metric_inputs: {metric_name}")


def check_primary_metric(
    checks: list[AuditCheck],
    *,
    manifest: dict[str, Any],
    route_data: dict[str, Any],
    rows: list[dict[str, str]],
    fields: set[str],
    schema: dict[str, str | None],
    run_dir: Path,
    gate: str,
) -> None:
    metric_name = route_primary_metric(route_data)
    if metric_name is None:
        return
    try:
        metric_inputs = manifest.get("metric_inputs")
        if isinstance(metric_inputs, dict) and isinstance(metric_inputs.get("score_matrix_csv"), str):
            score_matrix_path = resolve_from_root(metric_inputs["score_matrix_csv"], base=run_dir)
            recomputed = recompute_exact_metric_from_matrix(score_matrix_path, metric_name=metric_name, manifest=manifest)
        else:
            recomputed = recompute_primary_metric_from_predictions(metric_name=metric_name, rows=rows, fields=fields, schema=schema, manifest=manifest)
    except Exception as exc:
        add_warn_or_fail(
            checks,
            gate=gate,
            fail_gate=STRICT_GATES,
            rule_id="PRIMARY_METRIC_RECOMPUTE",
            message=f"primary metric could not be recomputed: {exc}",
            fix="Provide the prediction or metric_inputs artifacts required by the route primary metric.",
            decision_if_fail="DIAGNOSTIC_ONLY",
        )
        return

    add_check(checks, rule_id="PRIMARY_METRIC_RECOMPUTE", severity="INFO", status="PASS", message=f"recomputed {metric_name}: {recomputed:.6f}")
    compare_metric(checks, metric_name=metric_name, recomputed=recomputed, manifest=manifest)


def check_prediction_csv(path: Path, checks: list[AuditCheck], *, route_data: dict[str, Any], gate: str, manifest: dict[str, Any]) -> None:
    if not path.exists():
        add_check(
            checks,
            rule_id="PREDICTION_FILE_EXISTS",
            severity="ERROR",
            status="FAIL",
            message=f"prediction file is missing: {path}",
            fix="Write the prediction CSV listed in manifest.json or update the manifest.",
            decision_if_fail="DIAGNOSTIC_ONLY",
        )
        return

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = set(reader.fieldnames or [])

    if not rows:
        add_check(
            checks,
            rule_id="PREDICTION_NONEMPTY",
            severity="ERROR",
            status="FAIL",
            message="prediction CSV has no rows",
            fix="Regenerate predictions for the audited run.",
            decision_if_fail="DIAGNOSTIC_ONLY",
        )
        return
    add_check(checks, rule_id="PREDICTION_NONEMPTY", severity="INFO", status="PASS", message=f"prediction rows: {len(rows)}")

    schema = prediction_schema(fields)
    if schema["score"] is None:
        add_check(
            checks,
            rule_id="PREDICTION_SCORE_COLUMN",
            severity="WARN",
            status="WARN",
            message="prediction CSV has no standard score column",
            fix="Use one of: score, y_score, probability, logit.",
        )
    else:
        add_check(checks, rule_id="PREDICTION_SCORE_COLUMN", severity="INFO", status="PASS", message="score column found")

    top4_required = route_uses_top4(route_data)
    record_level = str(manifest.get("prediction_record_level") or "trial")
    if top4_required and record_level != "trial":
        add_warn_or_fail(
            checks,
            gate=gate,
            fail_gate={"diagnostic", "candidate", "promoted"},
            rule_id="PREDICTION_RECORD_LEVEL",
            message=f"Top-4 audit expects trial-level predictions, got: {record_level}",
            fix="Aggregate predictions to trial level before Top-4 audit, or set the correct route inference policy.",
            decision_if_fail="BLOCKED",
        )

    if schema["subject_id"] and schema["trial_id"] and schema["pred_top4"]:
        group_columns = top4_group_columns(fields, schema, manifest)
        if group_columns is None:
            add_warn_or_fail(
                checks,
                gate=gate,
                fail_gate={"diagnostic", "candidate", "promoted"} if top4_required else {"promoted"},
                rule_id="PREDICTION_TOP4_GROUP_KEYS",
                message="Top-4 group keys are not available in prediction CSV",
                fix="Set manifest top4_group_keys to columns such as subject_id, or include those columns.",
                decision_if_fail="BLOCKED",
            )
            group_columns = [schema["subject_id"]]

        bad_groups: list[str] = []
        groups: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            group_id = "|".join(str(row[col]) for col in group_columns)
            groups.setdefault(group_id, []).append(row)
        for group_id, group in groups.items():
            n_pred = sum(int(float(row[schema["pred_top4"]])) for row in group)
            if len(group) != 8 or n_pred != 4:
                bad_groups.append(f"{group_id}: rows={len(group)}, pred_top4={n_pred}")
        if bad_groups:
            add_check(
                checks,
                rule_id="PREDICTION_TOP4_GROUPS",
                severity="ERROR",
                status="FAIL",
                message="invalid Top-4 groups: " + "; ".join(bad_groups[:5]),
                fix="Ensure every subject has exactly 8 trials and exactly 4 positive Top-4 predictions.",
                decision_if_fail="DIAGNOSTIC_ONLY",
            )
        else:
            add_check(checks, rule_id="PREDICTION_TOP4_GROUPS", severity="INFO", status="PASS", message="Top-4 groups are valid")
        check_top4_group_semantics(checks, groups=groups, schema=schema)
    else:
        if not top4_required:
            add_check(
                checks,
                rule_id="PREDICTION_TOP4_GROUPS",
                severity="INFO",
                status="PASS",
                message="Top-4 audit is not required for this route",
            )
        else:
            add_warn_or_fail(
                checks,
                gate=gate,
                fail_gate={"diagnostic", "candidate", "promoted"},
                rule_id="PREDICTION_TOP4_GROUPS",
                message="Top-4 group columns are not present",
                fix="For Top-4 audits, include subject_id or user_id, trial_id, and pred_top4.",
                decision_if_fail="BLOCKED",
            )

    if schema["y_true"] and schema["y_pred"]:
        y_true = []
        y_pred = []
        for row in rows:
            y_true.append(int(float(row[schema["y_true"]])))
            y_pred.append(int(float(row[schema["y_pred"]])))
        score = balanced_accuracy(np.array(y_true), np.array(y_pred))
        add_check(checks, rule_id="METRIC_RECOMPUTE_BA", severity="INFO", status="PASS", message=f"recomputed BA: {score:.6f}")
    else:
        add_warn_or_fail(
            checks,
            gate=gate,
            fail_gate=STRICT_GATES,
            rule_id="METRIC_RECOMPUTE_BA",
            message="y_true/y_pred columns are not present; BA recompute skipped",
            fix="For labeled validation runs, include y_true and y_pred in the prediction CSV.",
            decision_if_fail="DIAGNOSTIC_ONLY",
        )

    check_primary_metric(checks, manifest=manifest, route_data=route_data, rows=rows, fields=fields, schema=schema, run_dir=path.parent, gate=gate)


def enforce_gate_strictness(checks: list[AuditCheck], *, gate: str) -> None:
    if gate in STRICT_GATES and any(check["status"] == "WARN" for check in checks):
        add_check(
            checks,
            rule_id="GATE_WARN_STRICTNESS",
            severity="ERROR",
            status="FAIL",
            message=f"{gate} gate does not allow unresolved WARN checks",
            fix="Resolve WARN checks or rerun under a less strict gate.",
            decision_if_fail="BLOCKED",
        )


def overall_decision(checks: list[AuditCheck], *, gate: str) -> str:
    enforce_gate_strictness(checks, gate=gate)
    for decision in ["REJECT", "BLOCKED", "DIAGNOSTIC_ONLY"]:
        if any(check["status"] == "FAIL" and check.get("decision_if_fail") == decision for check in checks):
            return decision
    if any(check["status"] == "WARN" for check in checks):
        return "WARN"
    return "PASS"


def write_reports(run_dir: Path, report: dict[str, Any]) -> None:
    json_path = run_dir / "audit_report.json"
    md_path = run_dir / "audit_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Experiment Audit Report",
        "",
        f"- route_id: `{report['route_id']}`",
        f"- gate: `{report['gate']}`",
        f"- audit_decision: `{report['overall']}`",
        f"- run_dir: `{report.get('run_dir') or ''}`",
        "",
        "| rule_id | severity | status | message | fix |",
        "|---|---|---|---|---|",
    ]
    for check in report["checks"]:
        message = str(check["message"]).replace("|", "\\|")
        fix = str(check.get("fix") or "").replace("|", "\\|")
        lines.append(f"| `{check['rule_id']}` | {check['severity']} | {check['status']} | {message} | {fix} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(route_path: Path, run_dir: Path | None, *, gate: str) -> dict[str, Any]:
    if gate not in GATES:
        raise ValueError(f"unknown gate: {gate}")
    checks: list[AuditCheck] = []
    route_path = route_path.resolve()

    try:
        route_data = load_yaml(route_path)
        route_errors = validate_route_config(route_data, path=route_path)
    except Exception as exc:
        route_data = {}
        route_errors = [str(exc)]

    route_id = str(route_data.get("route_id") or route_path.stem)
    if route_errors:
        add_check(
            checks,
            rule_id="ROUTE_SCHEMA",
            severity="CRITICAL",
            status="FAIL",
            message="; ".join(route_errors),
            fix="Fix the route config and rerun validate_route.py.",
            decision_if_fail="REJECT",
        )
    else:
        add_check(checks, rule_id="ROUTE_SCHEMA", severity="INFO", status="PASS", message="route config is valid")
        check_dataset_manifest(route_data, checks, gate=gate)
        check_split_manifest(route_data, checks, gate=gate)

    if run_dir is None:
        add_warn_or_fail(
            checks,
            gate=gate,
            fail_gate=STRICT_GATES,
            rule_id="RUN_DIRECTORY",
            message="no run directory provided; experiment artifact checks skipped",
            fix="Pass --run for a finished experiment audit.",
            decision_if_fail="BLOCKED",
        )
    else:
        run_dir = run_dir.resolve()
        if not run_dir.exists():
            add_check(
                checks,
                rule_id="RUN_DIRECTORY",
                severity="ERROR",
                status="FAIL",
                message=f"run directory is missing: {run_dir}",
                fix="Provide an existing run output directory.",
                decision_if_fail="BLOCKED",
            )
        else:
            add_check(checks, rule_id="RUN_DIRECTORY", severity="INFO", status="PASS", message="run directory exists")
            manifest_path = run_dir / "manifest.json"
            manifest_errors = validate_manifest(manifest_path, root=ROOT, route_data=route_data)
            if manifest_errors:
                decision = "DIAGNOSTIC_ONLY" if any("not found" in item for item in manifest_errors) else "BLOCKED"
                add_check(
                    checks,
                    rule_id="MANIFEST_VALID",
                    severity="ERROR",
                    status="FAIL",
                    message="; ".join(manifest_errors),
                    fix="Regenerate the run manifest from the actual route config and run artifacts.",
                    decision_if_fail=decision,
                )
            else:
                add_check(checks, rule_id="MANIFEST_VALID", severity="INFO", status="PASS", message="manifest is valid")
                manifest = load_manifest(manifest_path)
                prediction_csv = manifest.get("prediction_csv")
                if isinstance(prediction_csv, str) and prediction_csv:
                    check_prediction_csv(resolve_from_root(prediction_csv, base=run_dir), checks, route_data=route_data, gate=gate, manifest=manifest)
                else:
                    add_warn_or_fail(
                        checks,
                        gate=gate,
                        fail_gate=STRICT_GATES,
                        rule_id="PREDICTION_FILE_DECLARED",
                        message="manifest has no prediction_csv field",
                        fix="Add prediction_csv to manifest when predictions are available.",
                        decision_if_fail="BLOCKED",
                    )

            summary_path = ROOT / "reports" / "route_summaries" / f"{route_id}_summary.md"
            if summary_path.exists():
                add_check(checks, rule_id="SUMMARY_EXISTS", severity="INFO", status="PASS", message=f"summary found: {summary_path.relative_to(ROOT)}")
            else:
                add_warn_or_fail(
                    checks,
                    gate=gate,
                    fail_gate={"diagnostic", "candidate", "promoted"},
                    rule_id="SUMMARY_EXISTS",
                    message=f"route summary is missing: {summary_path.relative_to(ROOT)}",
                    fix="Write one concise route summary before treating this as a finished experiment.",
                    decision_if_fail="BLOCKED",
                )

    decision = overall_decision(checks, gate=gate)
    if decision not in AUDIT_DECISIONS:
        raise RuntimeError(f"unknown audit decision: {decision}")
    return {
        "route_id": route_id,
        "route_config": str(route_path),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "gate": gate,
        "overall": decision,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--gate", choices=sorted(GATES), default="candidate")
    args = parser.parse_args(argv)

    report = run_audit(args.route, args.run, gate=args.gate)
    if args.run is not None and args.run.exists():
        write_reports(args.run.resolve(), report)

    print(json.dumps({"route_id": report["route_id"], "gate": report["gate"], "overall": report["overall"]}, ensure_ascii=False))
    if report["overall"] in {"REJECT", "BLOCKED", "DIAGNOSTIC_ONLY"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
