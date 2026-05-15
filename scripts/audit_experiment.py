from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from hust_bci_er.audit.manifest import load_manifest, resolve_repo_or_run_path, sha256_file, validate_manifest  # noqa: E402
from hust_bci_er.audit.promotion import parse_key_value_markdown  # noqa: E402
from hust_bci_er.config.registry import AUDIT_DECISIONS  # noqa: E402
from hust_bci_er.config.schema import validate_route_config  # noqa: E402
from hust_bci_er.contracts.prediction import canonical_prediction_column, prediction_schema  # noqa: E402
from hust_bci_er.data.splits import assert_disjoint_subjects, assert_original_trial_not_cross_split  # noqa: E402
from hust_bci_er.evaluation.exact_single_crop import exact_all_correct_rate_from_matrix, exact_ba_from_matrix  # noqa: E402
from hust_bci_er.evaluation.metrics import balanced_accuracy  # noqa: E402
from hust_bci_er.inference.topk import topk_binary  # noqa: E402


AuditCheck = dict[str, Any]
GATES = {"smoke", "diagnostic", "candidate", "promoted"}
STRICT_GATES = {"candidate", "promoted"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PROMOTION_REQUIRED_FIELDS = {
    "route_id",
    "promoted_from_run",
    "candidate_audit_report",
    "primary_metric",
    "comparison_baseline",
    "risk_review",
    "no_leakage_review",
    "decision",
    "reviewer",
    "date",
}
PROMOTION_REQUIRED_CANDIDATE_RULES = {
    "MANIFEST_VALID",
    "PRIMARY_METRIC_RECOMPUTE",
    "PRIMARY_METRIC_REPORTED",
    "RUN_DATASET_EVIDENCE_VALID",
    "RUN_SPLIT_EVIDENCE_VALID",
    "RUN_SPLIT_EVIDENCE_CONSISTENT",
    "RUN_REPRODUCIBILITY_LOCKED",
}
PROMOTION_REQUIRED_TOP4_RULES = {
    "PREDICTION_TOP4_RANKING",
    "PREDICTION_TOP4_BINARY",
    "PREDICTION_TRIAL_ID_UNIQUE",
    "PREDICTION_TOP4_TRUTH_BALANCE",
}


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


def load_yaml_for_audit(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return None, str(exc)
    if not isinstance(data, dict):
        return None, "YAML document must be a mapping"
    return data, None


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def resolve_run_artifact(value: str, *, run_dir: Path, field: str) -> Path:
    resolved = (run_dir / value).resolve()
    if not is_under(resolved, run_dir):
        raise ValueError(f"{field} must stay under the run directory")
    return resolved


def parse_finite_float(value: Any, *, field: str, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric in {context}") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{field} is not finite in {context}")
    return parsed


def parse_binary(value: Any, *, field: str, context: str) -> int:
    parsed = parse_finite_float(value, field=field, context=context)
    if parsed not in {0.0, 1.0}:
        raise ValueError(f"{field} is not binary in {context}")
    return int(parsed)


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
        value = float(metrics[metric_name])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


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

    split_data, parse_error = load_yaml_for_audit(split_path)
    if parse_error is not None or split_data is None:
        add_check(
            checks,
            rule_id="SPLIT_MANIFEST_PARSE",
            severity="CRITICAL",
            status="FAIL",
            message=f"split manifest could not be parsed: {parse_error}",
            fix="Fix the split manifest YAML.",
            decision_if_fail="REJECT",
        )
        return
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

    add_check(checks, rule_id="SPLIT_MANIFEST_DECLARED", severity="INFO", status="PASS", message="route split manifest is declared")


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

    dataset_data, parse_error = load_yaml_for_audit(dataset_path)
    if parse_error is not None or dataset_data is None:
        add_check(
            checks,
            rule_id="DATASET_MANIFEST_PARSE",
            severity="CRITICAL",
            status="FAIL",
            message=f"dataset manifest could not be parsed: {parse_error}",
            fix="Fix the dataset manifest YAML.",
            decision_if_fail="REJECT",
        )
        return
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

    add_check(checks, rule_id="DATASET_MANIFEST_DECLARED", severity="INFO", status="PASS", message="route dataset manifest is declared")


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


def has_subject_membership(data: dict[str, Any]) -> bool:
    if not all(isinstance(data.get(f"{split}_subjects"), list) for split in ["train", "val", "test"]):
        return False
    return bool(data.get("train_subjects")) and bool(data.get("test_subjects"))


def split_manifest_has_evidence(split_data: dict[str, Any]) -> bool:
    has_subject_lists = has_subject_membership(split_data)
    folds = split_data.get("folds") or split_data.get("fold_definitions") or []
    has_fold_definitions = isinstance(folds, list) and bool(folds) and all(
        isinstance(fold, dict) and has_subject_membership(fold)
        for fold in folds
    )
    trial_rows = split_data.get("trial_rows") or []
    has_trial_rows = isinstance(trial_rows, list) and bool(trial_rows) and all(
        isinstance(row, dict) and {"subject_id", "original_trial_id", "split"}.issubset(row) for row in trial_rows
    )
    return has_subject_lists or has_fold_definitions or has_trial_rows


def split_manifest_has_formal_evidence(split_data: dict[str, Any]) -> bool:
    """Formal candidate evidence needs complete trial rows.

    Subject membership may be declared explicitly or derived from trial_rows,
    including per-fold rows.
    """
    trial_rows = split_data.get("trial_rows") or []
    has_trial_index = isinstance(trial_rows, list) and bool(trial_rows) and all(
        isinstance(row, dict) and {"subject_id", "original_trial_id", "split"}.issubset(row) for row in trial_rows
    )
    return has_trial_index and not trial_row_membership_errors([row for row in trial_rows if isinstance(row, dict)])


def subjects_from_trial_rows(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    subjects_by_split = {"train": set(), "val": set(), "test": set()}
    for row in rows:
        split = str(row.get("split"))
        subject = row.get("subject_id")
        if split in subjects_by_split and subject is not None:
            subjects_by_split[split].add(str(subject))
    return subjects_by_split


def trial_row_membership_errors(rows: list[dict[str, Any]]) -> list[str]:
    """Verify trial_rows derive usable train/test subject membership."""
    if not rows:
        return ["trial_rows must derive non-empty train and test subject membership"]
    has_fold_rows = any("fold" in row for row in rows)
    has_unfolded_rows = any("fold" not in row for row in rows)
    if has_fold_rows and has_unfolded_rows:
        return ["trial_rows must either all declare fold or none declare fold"]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("fold", "__single_fold__")), []).append(row)

    errors: list[str] = []
    for fold_id, fold_rows in sorted(grouped.items()):
        subjects = subjects_from_trial_rows(fold_rows)
        if not subjects["train"] or not subjects["test"]:
            if fold_id == "__single_fold__":
                errors.append("trial_rows must derive non-empty train and test subject membership")
            else:
                errors.append(f"trial_rows fold {fold_id} must derive non-empty train and test subject membership")
    return errors


def split_evidence_consistency_errors(split_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rows = split_data.get("trial_rows") or []
    if not isinstance(rows, list) or not rows:
        return ["trial_rows must be a non-empty list"]
    valid_splits = {"train", "val", "test"}
    normalized_rows = [row for row in rows if isinstance(row, dict)]
    if len(normalized_rows) != len(rows):
        errors.append("trial_rows must contain only mappings")
    for idx, row in enumerate(normalized_rows):
        split = row.get("split")
        subject = row.get("subject_id")
        original_trial = row.get("original_trial_id")
        if split not in valid_splits:
            errors.append(f"trial_rows[{idx}].split must be train/val/test")
        if subject in {None, ""}:
            errors.append(f"trial_rows[{idx}].subject_id is required")
        if original_trial in {None, ""}:
            errors.append(f"trial_rows[{idx}].original_trial_id is required")

    consumed: set[int] = set()
    indexed_rows = list(enumerate(normalized_rows))
    folds = split_data.get("folds") or split_data.get("fold_definitions") or []
    has_declared_folds = isinstance(folds, list) and bool(folds)
    rows_with_fold = [idx for idx, row in indexed_rows if "fold" in row]
    rows_without_fold = [idx for idx, row in indexed_rows if "fold" not in row]
    if rows_with_fold and rows_without_fold:
        errors.append("trial_rows must either all declare fold or none declare fold")
    errors.extend(trial_row_membership_errors(normalized_rows))

    if has_subject_membership(split_data):
        top_level_rows = [(idx, row) for idx, row in indexed_rows if "fold" not in row]
        if not top_level_rows:
            errors.append("top-level subject lists require trial_rows without fold")
        else:
            consumed.update(idx for idx, _ in top_level_rows)
            row_subjects = subjects_from_trial_rows([row for _, row in top_level_rows])
            for split in ["train", "val", "test"]:
                expected = set(map(str, split_data.get(f"{split}_subjects") or []))
                actual = row_subjects[split]
                if expected != actual:
                    missing = sorted(expected - actual)[:5]
                    extra = sorted(actual - expected)[:5]
                    errors.append(f"top-level {split}_subjects do not match trial_rows; missing={missing}, extra={extra}")

    declared_fold_ids: set[str] = set()
    if has_declared_folds:
        for idx, fold in enumerate(folds):
            if not isinstance(fold, dict):
                errors.append(f"fold {idx} must be a mapping")
                continue
            fold_id = fold.get("fold", fold.get("fold_id", idx))
            declared_fold_ids.add(str(fold_id))
            fold_rows = [(row_idx, row) for row_idx, row in indexed_rows if str(row.get("fold")) == str(fold_id)]
            if not fold_rows:
                errors.append(f"fold {fold_id} has no trial_rows")
                continue
            consumed.update(row_idx for row_idx, _ in fold_rows)
            row_subjects = subjects_from_trial_rows([row for _, row in fold_rows])
            for split in ["train", "val", "test"]:
                expected = set(map(str, fold.get(f"{split}_subjects") or []))
                actual = row_subjects[split]
                if expected != actual:
                    missing = sorted(expected - actual)[:5]
                    extra = sorted(actual - expected)[:5]
                    errors.append(f"fold {fold_id} {split}_subjects do not match trial_rows; missing={missing}, extra={extra}")
        for idx, row in indexed_rows:
            if "fold" not in row:
                errors.append(f"trial_rows[{idx}] must declare fold when fold definitions are present")
            elif str(row.get("fold")) not in declared_fold_ids:
                errors.append(f"trial_rows[{idx}].fold is not declared")

    unconsumed = sorted(set(range(len(normalized_rows))) - consumed)
    if unconsumed and (has_subject_membership(split_data) or has_declared_folds):
        errors.append("trial_rows contain rows not covered by subject lists or fold definitions: " + ", ".join(map(str, unconsumed[:5])))
    return errors


def subject_sets_from_split(data: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    return (
        set(map(str, data.get("train_subjects") or [])),
        set(map(str, data.get("val_subjects") or [])),
        set(map(str, data.get("test_subjects") or [])),
    )


def split_leakage_errors(split_data: dict[str, Any]) -> tuple[list[str], list[str]]:
    subject_errors: list[str] = []
    trial_errors: list[str] = []

    train_subjects, val_subjects, test_subjects = subject_sets_from_split(split_data)
    if train_subjects or val_subjects or test_subjects:
        try:
            assert_disjoint_subjects(train_subjects, val_subjects, test_subjects)
        except ValueError as exc:
            subject_errors.append(f"top-level: {exc}")

    folds = split_data.get("folds") or split_data.get("fold_definitions") or []
    if isinstance(folds, list):
        for idx, fold in enumerate(folds):
            if not isinstance(fold, dict):
                continue
            try:
                assert_disjoint_subjects(*subject_sets_from_split(fold))
            except ValueError as exc:
                subject_errors.append(f"fold {idx}: {exc}")

    rows = split_data.get("trial_rows") or []
    if isinstance(rows, list) and rows:
        rows_by_fold: dict[str, list[dict[str, Any]]] = {}
        has_fold_rows = any(isinstance(row, dict) and "fold" in row for row in rows)
        has_unfolded_rows = any(isinstance(row, dict) and "fold" not in row for row in rows)
        if has_fold_rows and has_unfolded_rows:
            subject_errors.append("trial_rows mix folded and unfolded rows")
            trial_errors.append("trial_rows mix folded and unfolded rows")
        if not has_fold_rows:
            try:
                assert_original_trial_not_cross_split(rows)
            except (KeyError, ValueError) as exc:
                trial_errors.append(str(exc))
        for row in rows:
            if not isinstance(row, dict):
                continue
            rows_by_fold.setdefault(str(row.get("fold", "__single_fold__")), []).append(row)
        for fold, fold_rows in rows_by_fold.items():
            if has_fold_rows:
                try:
                    assert_original_trial_not_cross_split(fold_rows)
                except (KeyError, ValueError) as exc:
                    trial_errors.append(f"fold {fold}: {exc}")
            subjects_by_split = {"train": set(), "val": set(), "test": set()}
            for row in fold_rows:
                split = str(row.get("split"))
                subject = row.get("subject_id")
                if split in subjects_by_split and subject is not None:
                    subjects_by_split[split].add(str(subject))
            try:
                assert_disjoint_subjects(subjects_by_split["train"], subjects_by_split["val"], subjects_by_split["test"])
            except ValueError as exc:
                subject_errors.append(f"trial_rows fold {fold}: {exc}")

    return subject_errors, trial_errors


def add_split_leakage_checks(checks: list[AuditCheck], split_data: dict[str, Any], *, gate: str) -> None:
    subject_errors, trial_errors = split_leakage_errors(split_data)
    if subject_errors:
        add_check(
            checks,
            rule_id="RUN_SPLIT_SUBJECT_DISJOINT",
            severity="ERROR",
            status="FAIL",
            message="run split subject overlap detected: " + "; ".join(subject_errors[:3]),
            fix="Regenerate split evidence so train/val/test subjects are disjoint in every split definition.",
            decision_if_fail="REJECT",
        )
    else:
        add_check(checks, rule_id="RUN_SPLIT_SUBJECT_DISJOINT", severity="INFO", status="PASS", message="run split subjects are disjoint")

    if trial_errors:
        add_check(
            checks,
            rule_id="RUN_SPLIT_ORIGINAL_TRIAL_DISJOINT",
            severity="ERROR",
            status="FAIL",
            message="run split original-trial leakage detected: " + "; ".join(trial_errors[:3]),
            fix="Regenerate split evidence so each original trial belongs to exactly one split.",
            decision_if_fail="REJECT",
        )
    else:
        add_check(checks, rule_id="RUN_SPLIT_ORIGINAL_TRIAL_DISJOINT", severity="INFO", status="PASS", message="run split original trials do not cross splits")


def dataset_manifest_has_evidence(dataset_data: dict[str, Any]) -> bool:
    data_sources = dataset_data.get("data_sources") or []
    checksum_manifest = dataset_data.get("checksum_manifest") or []
    has_data_sources = isinstance(data_sources, list) and bool(data_sources) and all(
        isinstance(item, dict) and item.get("path") and item.get("kind") for item in data_sources
    )
    has_checksums = isinstance(checksum_manifest, list) and bool(checksum_manifest) and all(
        isinstance(item, dict) and item.get("path") and item.get("sha256") for item in checksum_manifest
    )
    return has_data_sources and has_checksums


def dataset_label_scope_errors(dataset_data: dict[str, Any]) -> list[str]:
    scope = dataset_data.get("label_scope")
    if not isinstance(scope, dict):
        return ["label_scope must be a mapping"]
    allowed = {"available", "available_for_audit_only", "hidden", "hidden_until_audit", "not_applicable"}
    required = {"train", "val", "test", "pseudo_public"}
    errors: list[str] = []
    missing = sorted(required - set(scope))
    if missing:
        errors.append("label_scope missing keys: " + ", ".join(missing))
    for key, value in sorted(scope.items()):
        if key not in required:
            errors.append(f"label_scope.{key} is not a known split scope")
        elif value not in allowed:
            errors.append(f"label_scope.{key} has unsupported value: {value}")
    if scope.get("pseudo_public") == "available":
        errors.append("label_scope.pseudo_public must not be available")
    if scope.get("test") == "available":
        errors.append("label_scope.test should be hidden or available_for_audit_only")
    return errors


def dataset_checksum_errors(dataset_data: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    schema_errors: list[str] = []
    coverage_errors: list[str] = []
    extra_errors: list[str] = []
    data_sources = dataset_data.get("data_sources") or []
    checksum_manifest = dataset_data.get("checksum_manifest") or []
    source_paths: list[str] = []
    checksum_paths: list[str] = []

    if isinstance(data_sources, list) and data_sources:
        for idx, item in enumerate(data_sources):
            if not isinstance(item, dict):
                schema_errors.append(f"data_sources[{idx}] must be a mapping")
                continue
            path = item.get("path")
            if not isinstance(path, str) or not path:
                schema_errors.append(f"data_sources[{idx}].path is required")
            else:
                source_paths.append(path)
            if not item.get("kind"):
                schema_errors.append(f"data_sources[{idx}].kind is required")
    else:
        schema_errors.append("data_sources must be a non-empty list")

    if isinstance(checksum_manifest, list) and checksum_manifest:
        for idx, item in enumerate(checksum_manifest):
            if not isinstance(item, dict):
                schema_errors.append(f"checksum_manifest[{idx}] must be a mapping")
                continue
            path = item.get("path")
            checksum = item.get("sha256")
            if not isinstance(path, str) or not path:
                schema_errors.append(f"checksum_manifest[{idx}].path is required")
            else:
                checksum_paths.append(path)
            if not isinstance(checksum, str) or SHA256_RE.fullmatch(checksum) is None:
                schema_errors.append(f"checksum_manifest[{idx}].sha256 must be lowercase sha256")
    else:
        schema_errors.append("checksum_manifest must be a non-empty list")

    duplicate_sources = sorted(path for path in set(source_paths) if source_paths.count(path) > 1)
    duplicate_checksums = sorted(path for path in set(checksum_paths) if checksum_paths.count(path) > 1)
    if duplicate_sources:
        schema_errors.append("duplicate data_sources paths: " + ", ".join(duplicate_sources[:5]))
    if duplicate_checksums:
        schema_errors.append("duplicate checksum_manifest paths: " + ", ".join(duplicate_checksums[:5]))

    missing_checksums = sorted(set(source_paths) - set(checksum_paths))
    if missing_checksums:
        coverage_errors.append("checksum_manifest does not cover data_sources: " + ", ".join(missing_checksums[:5]))
    extra_checksums = sorted(set(checksum_paths) - set(source_paths))
    if extra_checksums:
        extra_errors.append("checksum_manifest has paths not declared in data_sources: " + ", ".join(extra_checksums[:5]))
    return schema_errors, coverage_errors, extra_errors


def add_dataset_checksum_checks(checks: list[AuditCheck], dataset_data: dict[str, Any], *, gate: str) -> None:
    schema_errors, coverage_errors, extra_errors = dataset_checksum_errors(dataset_data)
    if schema_errors:
        add_warn_or_fail(
            checks,
            gate=gate,
            fail_gate=STRICT_GATES,
            rule_id="RUN_DATASET_CHECKSUM_SCHEMA",
            message="dataset checksum schema is invalid: " + "; ".join(schema_errors[:5]),
            fix="Use unique paths and lowercase 64-hex sha256 values in checksum_manifest.",
            decision_if_fail="BLOCKED",
        )
    else:
        add_check(checks, rule_id="RUN_DATASET_CHECKSUM_SCHEMA", severity="INFO", status="PASS", message="dataset checksum schema is valid")

    if coverage_errors:
        add_warn_or_fail(
            checks,
            gate=gate,
            fail_gate=STRICT_GATES,
            rule_id="RUN_DATASET_CHECKSUM_COVERAGE",
            message="dataset checksum coverage is incomplete: " + "; ".join(coverage_errors[:5]),
            fix="Add checksum_manifest entries for every data_sources path.",
            decision_if_fail="BLOCKED",
        )
    else:
        add_check(checks, rule_id="RUN_DATASET_CHECKSUM_COVERAGE", severity="INFO", status="PASS", message="dataset checksums cover all data sources")

    if extra_errors:
        add_warn_or_fail(
            checks,
            gate=gate,
            fail_gate=STRICT_GATES,
            rule_id="RUN_DATASET_CHECKSUM_EXTRA",
            message="dataset checksum manifest has extra paths: " + "; ".join(extra_errors[:5]),
            fix="Remove stale checksum entries or add matching data_sources entries.",
            decision_if_fail="BLOCKED",
        )
    else:
        add_check(checks, rule_id="RUN_DATASET_CHECKSUM_EXTRA", severity="INFO", status="PASS", message="dataset checksum manifest has no extra paths")


def add_dataset_label_scope_check(checks: list[AuditCheck], dataset_data: dict[str, Any], *, gate: str) -> None:
    errors = dataset_label_scope_errors(dataset_data)
    if errors:
        add_warn_or_fail(
            checks,
            gate=gate,
            fail_gate=STRICT_GATES,
            rule_id="RUN_DATASET_LABEL_SCOPE",
            message="dataset label scope is invalid: " + "; ".join(errors[:5]),
            fix="Declare label_scope for train/val/test/pseudo_public and keep public inference labels hidden or audit-only.",
            decision_if_fail="BLOCKED",
        )
    else:
        add_check(checks, rule_id="RUN_DATASET_LABEL_SCOPE", severity="INFO", status="PASS", message="dataset label scope is explicit")


def check_run_dataset_split_evidence(
    manifest: dict[str, Any],
    route_data: dict[str, Any],
    run_dir: Path,
    checks: list[AuditCheck],
    *,
    gate: str,
) -> None:
    errors: list[str] = []
    dataset_path_value = manifest.get("dataset_manifest_path")
    split_path_value = manifest.get("split_manifest_path")

    if isinstance(dataset_path_value, str):
        dataset_path = resolve_repo_or_run_path(
            run_dir,
            ROOT,
            dataset_path_value,
            field="dataset_manifest_path",
            repo_subdir="configs/datasets",
            errors=errors,
        )
        if dataset_path is None or not dataset_path.exists():
            add_check(
                checks,
                rule_id="RUN_DATASET_EVIDENCE_VALID",
                severity="ERROR",
                status="FAIL",
                message="dataset evidence path is invalid: " + "; ".join(errors),
                fix="Point manifest.dataset_manifest_path to a run-local or configs/datasets evidence file.",
                decision_if_fail="BLOCKED",
            )
        else:
            dataset_data, parse_error = load_yaml_for_audit(dataset_path)
            expected_dataset = route_data.get("dataset_version")
            if parse_error is not None or dataset_data is None:
                add_check(
                    checks,
                    rule_id="RUN_DATASET_EVIDENCE_PARSE",
                    severity="ERROR",
                    status="FAIL",
                    message=f"run dataset evidence could not be parsed: {parse_error}",
                    fix="Fix the dataset evidence YAML.",
                    decision_if_fail="BLOCKED",
                )
            elif expected_dataset and dataset_data.get("dataset_version") != expected_dataset:
                add_check(
                    checks,
                    rule_id="RUN_DATASET_EVIDENCE_VALID",
                    severity="ERROR",
                    status="FAIL",
                    message="run dataset evidence dataset_version does not match route",
                    fix="Use dataset evidence for the audited route dataset_version.",
                    decision_if_fail="BLOCKED",
                )
            elif dataset_data.get("status") == "declared_without_raw_data_index" or not dataset_manifest_has_evidence(dataset_data):
                add_warn_or_fail(
                    checks,
                    gate=gate,
                    fail_gate=STRICT_GATES,
                    rule_id="RUN_DATASET_EVIDENCE_VALID",
                    message="run dataset evidence is not verifiable",
                    fix="Add non-empty data_sources and checksum_manifest evidence before candidate/promoted review.",
                    decision_if_fail="BLOCKED",
                )
            else:
                add_check(checks, rule_id="RUN_DATASET_EVIDENCE_VALID", severity="INFO", status="PASS", message="run dataset evidence is verifiable")
            if dataset_data is not None:
                add_dataset_label_scope_check(checks, dataset_data, gate=gate)
                add_dataset_checksum_checks(checks, dataset_data, gate=gate)

    errors = []
    if isinstance(split_path_value, str):
        split_path = resolve_repo_or_run_path(
            run_dir,
            ROOT,
            split_path_value,
            field="split_manifest_path",
            repo_subdir="configs/splits",
            errors=errors,
        )
        if split_path is None or not split_path.exists():
            add_check(
                checks,
                rule_id="RUN_SPLIT_EVIDENCE_VALID",
                severity="ERROR",
                status="FAIL",
                message="split evidence path is invalid: " + "; ".join(errors),
                fix="Point manifest.split_manifest_path to a run-local or configs/splits evidence file.",
                decision_if_fail="BLOCKED",
            )
        else:
            split_data, parse_error = load_yaml_for_audit(split_path)
            expected_split = manifest.get("split_id") or route_data.get("split_id")
            if parse_error is not None or split_data is None:
                add_check(
                    checks,
                    rule_id="RUN_SPLIT_EVIDENCE_PARSE",
                    severity="ERROR",
                    status="FAIL",
                    message=f"run split evidence could not be parsed: {parse_error}",
                    fix="Fix the split evidence YAML.",
                    decision_if_fail="BLOCKED",
                )
            elif expected_split and split_data.get("split_id") != expected_split:
                add_check(
                    checks,
                    rule_id="RUN_SPLIT_EVIDENCE_VALID",
                    severity="ERROR",
                    status="FAIL",
                    message="run split evidence split_id does not match manifest",
                    fix="Use split evidence for the audited manifest split_id.",
                    decision_if_fail="BLOCKED",
                )
            elif split_data.get("status") == "declared_without_subject_list" or not split_manifest_has_formal_evidence(split_data):
                add_warn_or_fail(
                    checks,
                    gate=gate,
                    fail_gate=STRICT_GATES,
                    rule_id="RUN_SPLIT_EVIDENCE_VALID",
                    message="run split evidence is not verifiable",
                    fix="Add complete trial_rows with subject_id, original_trial_id, and split before candidate/promoted review; subject membership may be derived from those rows.",
                    decision_if_fail="BLOCKED",
                )
            else:
                add_check(checks, rule_id="RUN_SPLIT_EVIDENCE_VALID", severity="INFO", status="PASS", message="run split evidence is verifiable")
                consistency_errors = split_evidence_consistency_errors(split_data)
                if consistency_errors:
                    add_check(
                        checks,
                        rule_id="RUN_SPLIT_EVIDENCE_CONSISTENT",
                        severity="ERROR",
                        status="FAIL",
                        message="run split evidence is inconsistent: " + "; ".join(consistency_errors[:5]),
                        fix="Make subject lists or fold definitions match trial_rows exactly, and use only train/val/test split values.",
                        decision_if_fail="BLOCKED",
                    )
                else:
                    add_check(checks, rule_id="RUN_SPLIT_EVIDENCE_CONSISTENT", severity="INFO", status="PASS", message="run split subject lists match trial_rows")
            if split_data is not None and split_data.get("status") != "declared_without_subject_list" and split_manifest_has_evidence(split_data):
                add_split_leakage_checks(checks, split_data, gate=gate)


def check_reproducibility_manifest(manifest: dict[str, Any], checks: list[AuditCheck], *, gate: str, root: Path = ROOT) -> None:
    missing: list[str] = []
    malformed: list[str] = []

    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        missing.append("environment")
    else:
        if not isinstance(environment.get("python"), dict) or not environment["python"].get("version"):
            malformed.append("environment.python.version")
        if not isinstance(environment.get("packages"), dict):
            malformed.append("environment.packages")
        if not isinstance(environment.get("torch"), dict):
            malformed.append("environment.torch")

    determinism = manifest.get("determinism")
    required_determinism = {
        "python_seed",
        "python_hash_seed",
        "pythonhashseed_env",
        "numpy_seed",
        "torch_seed",
        "deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "dataloader_worker_seed_base",
        "batch_order",
    }
    if not isinstance(determinism, dict):
        missing.append("determinism")
    else:
        absent = sorted(key for key in required_determinism if key not in determinism)
        malformed.extend(f"determinism.{key}" for key in absent)
        manifest_seed = manifest.get("seed")
        for key in ["python_seed", "python_hash_seed", "numpy_seed", "torch_seed", "dataloader_worker_seed_base"]:
            if key in determinism and type(determinism[key]) is not int:
                malformed.append(f"determinism.{key} must be int")
            elif key in determinism and type(manifest_seed) is int and determinism[key] != manifest_seed:
                malformed.append(f"determinism.{key} must match manifest seed")
        if "pythonhashseed_env" in determinism:
            if determinism["pythonhashseed_env"] in {None, ""}:
                malformed.append("determinism.pythonhashseed_env must be set before Python starts")
            elif type(manifest_seed) is int and str(determinism["pythonhashseed_env"]) != str(manifest_seed):
                malformed.append("determinism.pythonhashseed_env must match manifest seed")
        if "seed" in determinism and type(manifest_seed) is int and determinism["seed"] != manifest_seed:
            malformed.append("determinism.seed must match manifest seed")
        if determinism.get("deterministic_algorithms") is not True:
            malformed.append("determinism.deterministic_algorithms must be true")
        if determinism.get("cudnn_deterministic") is not True:
            malformed.append("determinism.cudnn_deterministic must be true")
        if determinism.get("cudnn_benchmark") is not False:
            malformed.append("determinism.cudnn_benchmark must be false")
        batch_order = determinism.get("batch_order")
        if not isinstance(batch_order, dict):
            malformed.append("determinism.batch_order")
        else:
            if batch_order.get("policy") != "seeded_sampler_or_shuffle_false":
                malformed.append("determinism.batch_order.policy")
            if type(manifest_seed) is int and batch_order.get("sampler_seed") != manifest_seed:
                malformed.append("determinism.batch_order.sampler_seed")

    checkpoint_selection = manifest.get("checkpoint_selection")
    if not isinstance(checkpoint_selection, dict):
        missing.append("checkpoint_selection")
    else:
        if checkpoint_selection.get("rule") != "best_monitored_epoch":
            malformed.append("checkpoint_selection.rule")
        if checkpoint_selection.get("monitor") not in {"train_loss", "train_accuracy", "val_loss", "val_accuracy"}:
            malformed.append("checkpoint_selection.monitor")
        if checkpoint_selection.get("mode") not in {"min", "max"}:
            malformed.append("checkpoint_selection.mode")
        if checkpoint_selection.get("tie_break") != "earliest_epoch":
            malformed.append("checkpoint_selection.tie_break")
        if checkpoint_selection.get("restore_best") is not True:
            malformed.append("checkpoint_selection.restore_best")

    crop_policy = manifest.get("crop_policy")
    if not isinstance(crop_policy, dict):
        missing.append("crop_policy")
    else:
        for key in ["name", "selection", "tie_break"]:
            if not crop_policy.get(key):
                malformed.append(f"crop_policy.{key}")
        if crop_policy.get("name") == "random":
            if type(crop_policy.get("random_seed")) is not int:
                malformed.append("crop_policy.random_seed")
            elif type(manifest.get("seed")) is int and crop_policy.get("random_seed") != manifest["seed"]:
                malformed.append("crop_policy.random_seed must match manifest seed")
            if crop_policy.get("selection") != "per_trial_uniform_crop":
                malformed.append("crop_policy.selection")
        if crop_policy.get("name") == "worst":
            if crop_policy.get("selection") != "label_aware_min_metric_stress_test":
                malformed.append("crop_policy.selection")
            if crop_policy.get("tie_break") != "lowest_assignment_index":
                malformed.append("crop_policy.tie_break")

    environment_lock = manifest.get("environment_lock")
    if not isinstance(environment_lock, dict):
        missing.append("environment_lock")
    else:
        files = environment_lock.get("files")
        if not isinstance(files, list) or not files:
            malformed.append("environment_lock.files")
        else:
            paths = {str(item.get("path")) for item in files if isinstance(item, dict)}
            if "environment.lock" not in paths:
                malformed.append("environment_lock.environment.lock")
            if "requirements.lock" not in paths:
                malformed.append("environment_lock.requirements.lock")
            for item in files:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str) or SHA256_RE.fullmatch(str(item.get("sha256") or "")) is None:
                    malformed.append("environment_lock.file")
                    break
                lock_path = (root / item["path"]).resolve()
                if not is_under(lock_path, root):
                    malformed.append(f"environment_lock.{item['path']} escapes repository")
                elif not lock_path.exists():
                    malformed.append(f"environment_lock.{item['path']} missing")
                elif sha256_file(lock_path) != item["sha256"]:
                    malformed.append(f"environment_lock.{item['path']} sha256")

    if missing or malformed:
        add_warn_or_fail(
            checks,
            gate=gate,
            fail_gate=STRICT_GATES,
            rule_id="RUN_REPRODUCIBILITY_LOCKED",
            message="run reproducibility metadata is incomplete: " + ", ".join(missing + malformed),
            fix="Regenerate manifest.json with environment, determinism, checkpoint_selection, crop_policy, and environment_lock entries.",
            decision_if_fail="BLOCKED",
        )
    else:
        add_check(checks, rule_id="RUN_REPRODUCIBILITY_LOCKED", severity="INFO", status="PASS", message="run reproducibility metadata is locked")


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
    gate: str,
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
    bad_score: list[str] = []
    bad_ranking: list[str] = []
    for group_id, group in groups.items():
        trial_ids = [str(row[trial_col]) for row in group]
        if len(set(trial_ids)) != len(trial_ids):
            bad_unique.append(group_id)

        observed: list[int] = []
        for row in group:
            try:
                observed.append(parse_binary(row[top4_col], field=top4_col, context=group_id))
            except ValueError:
                bad_binary.append(group_id)
                observed.append(-1)

        if truth_col is not None:
            truth_values: list[int] = []
            for row in group:
                try:
                    truth_values.append(parse_binary(row[truth_col], field=truth_col, context=group_id))
                except ValueError:
                    bad_truth.append(group_id)
                    truth_values.append(-1)
            if len(group) == 8 and sum(truth_values) != 4:
                bad_truth.append(group_id)

        if score_col is not None and len(group) == 8 and all(value in {0, 1} for value in observed):
            parsed_scores: list[float] = []
            for row in group:
                try:
                    parsed_scores.append(parse_finite_float(row[score_col], field=score_col, context=group_id))
                except ValueError:
                    bad_score.append(group_id)
                    parsed_scores = []
                    break
            if not parsed_scores:
                continue
            scores = np.array(parsed_scores, dtype=float)
            expected = topk_binary(scores, 4)
            if not np.array_equal(expected, np.array(observed, dtype=int)):
                bad_ranking.append(group_id)

    semantic_specs = [
        ("PREDICTION_TRIAL_ID_UNIQUE", bad_unique, "duplicate trial_id values inside Top-4 groups", "Ensure each Top-4 group has 8 unique trial_id values."),
        ("PREDICTION_TOP4_BINARY", bad_binary, "pred_top4 contains non-binary values", "Encode pred_top4 as 0 or 1 only."),
    ]
    if truth_col is None:
        add_warn_or_fail(
            checks,
            gate=gate,
            fail_gate=STRICT_GATES,
            rule_id="PREDICTION_TOP4_TRUTH_BALANCE",
            message="truth balance was not checked because y_true is absent",
            fix="For Top-4 candidate/promoted audits, include y_true in prediction CSV or provide an explicitly supported truth-balance evidence path.",
            decision_if_fail="BLOCKED",
        )
    else:
        semantic_specs.append(("PREDICTION_TOP4_TRUTH_BALANCE", bad_truth, "y_true is not binary or not 4 positives per 8-trial group", "For labeled Top-4 audits, each 8-trial group should contain 4 positive labels."))

    if score_col is None:
        add_check(
            checks,
            rule_id="PREDICTION_SCORE_NUMERIC",
            severity="WARN",
            status="WARN",
            message="score numeric check was not run because no score column is present",
            fix="Include score, y_score, probability, or logit so Top-4 ranking can be recomputed.",
        )
        add_check(
            checks,
            rule_id="PREDICTION_TOP4_RANKING",
            severity="WARN",
            status="WARN",
            message="Top-4 ranking was not checked because no score column is present",
            fix="Include score, y_score, probability, or logit so pred_top4 can be recomputed.",
        )
    else:
        semantic_specs.extend(
            [
                ("PREDICTION_SCORE_NUMERIC", bad_score, "score contains non-numeric or non-finite values", "Encode score columns as finite numeric values."),
                ("PREDICTION_TOP4_RANKING", bad_ranking, "pred_top4 does not match score-derived top 4", "Regenerate pred_top4 from score using the repository Top-4 policy."),
            ]
        )

    for rule_id, bad, message, fix in semantic_specs:
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


def add_group_check(checks: list[AuditCheck], *, rule_id: str, bad: list[str], message: str, fix: str) -> None:
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
        add_check(checks, rule_id=rule_id, severity="INFO", status="PASS", message=f"{rule_id} passed")


def recompute_exact_metric_from_matrix(path: Path, *, metric_name: str, manifest: dict[str, Any], checks: list[AuditCheck]) -> float:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fields = set(reader.fieldnames or [])
    except Exception as exc:
        raise ValueError(f"score matrix CSV could not be parsed: {exc}") from exc
    schema = prediction_schema(fields)
    group_cols = metric_group_columns(fields, schema, manifest)
    score_cols = crop_score_columns(fields)
    truth_col = schema["y_true"]
    trial_col = schema["trial_id"]
    if group_cols is None or truth_col is None or trial_col is None or len(score_cols) != 5:
        raise ValueError("score matrix CSV must include group keys, trial_id, y_true, and crop_0..crop_4 columns")
    groups = group_prediction_rows(rows, group_cols)
    values: list[float] = []
    bad_group_size: list[str] = []
    bad_trial_unique: list[str] = []
    bad_label_binary: list[str] = []
    bad_truth_balance: list[str] = []
    bad_score_numeric: list[str] = []
    for group_id, group in groups.items():
        if len(group) != 8:
            bad_group_size.append(group_id)
            continue
        trial_ids = [str(row[trial_col]) for row in group]
        if len(set(trial_ids)) != len(trial_ids):
            bad_trial_unique.append(group_id)

        score_rows: list[list[float]] = []
        for row in group:
            parsed_row: list[float] = []
            for col in score_cols:
                try:
                    parsed_row.append(parse_finite_float(row[col], field=col, context=group_id))
                except ValueError:
                    bad_score_numeric.append(group_id)
                    parsed_row = []
                    break
            if not parsed_row:
                break
            score_rows.append(parsed_row)
        if len(score_rows) != len(group):
            continue

        labels: list[int] = []
        for row in group:
            try:
                labels.append(parse_binary(row[truth_col], field=truth_col, context=group_id))
            except ValueError:
                bad_label_binary.append(group_id)
                labels.append(-1)
        if any(label not in {0, 1} for label in labels):
            continue
        if sum(labels) != 4:
            bad_truth_balance.append(group_id)
            continue

        mat = np.array(score_rows, dtype=float)
        y_true = np.array(labels, dtype=int)
        if metric_name == "exact_single_crop_expected_BA":
            values.append(exact_ba_from_matrix(mat, y_true))
        elif metric_name == "all_correct_rate":
            values.append(exact_all_correct_rate_from_matrix(mat, y_true))
        else:
            raise ValueError(f"unsupported exact metric: {metric_name}")
    add_group_check(checks, rule_id="SCORE_MATRIX_GROUP_SIZE", bad=bad_group_size, message="score matrix groups must contain 8 trials", fix="Write one row per trial for each metric group.")
    add_group_check(checks, rule_id="SCORE_MATRIX_TRIAL_ID_UNIQUE", bad=bad_trial_unique, message="score matrix groups contain duplicate trial_id values", fix="Ensure each score matrix group has 8 unique trial IDs.")
    add_group_check(checks, rule_id="SCORE_MATRIX_LABEL_BINARY", bad=bad_label_binary, message="score matrix y_true contains non-binary values", fix="Encode score matrix y_true as 0 or 1.")
    add_group_check(checks, rule_id="SCORE_MATRIX_TRUTH_BALANCE", bad=bad_truth_balance, message="score matrix groups do not contain 4 positive labels", fix="Ensure every 8-trial score matrix group has exactly 4 positives.")
    add_group_check(checks, rule_id="SCORE_MATRIX_SCORE_NUMERIC", bad=bad_score_numeric, message="score matrix contains non-numeric or non-finite crop scores", fix="Encode crop scores as finite numeric values.")
    if bad_group_size or bad_trial_unique or bad_label_binary or bad_truth_balance or bad_score_numeric:
        raise ValueError("score matrix semantic checks failed")
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
        pred_col = schema["pred_top4"] if metric_name == "top4_BA" else schema["y_pred"]
        if schema["y_true"] is None or pred_col is None:
            required = "y_true and pred_top4" if metric_name == "top4_BA" else "y_true and y_pred"
            raise ValueError(f"prediction CSV must include {required}")
        group_cols = metric_group_columns(fields, schema, manifest)
        if group_cols is None:
            raise ValueError("metric group keys are missing")
        values: list[float] = []
        for group_id, group in group_prediction_rows(rows, group_cols).items():
            y_true = np.array([parse_binary(row[schema["y_true"]], field=schema["y_true"], context=group_id) for row in group], dtype=int)
            y_pred = np.array([parse_binary(row[pred_col], field=pred_col, context=group_id) for row in group], dtype=int)
            value = balanced_accuracy(y_true, y_pred)
            if not np.isfinite(value):
                raise ValueError(f"primary metric is not finite for group {group_id}")
            values.append(value)
        if not values:
            raise ValueError("prediction CSV has no metric groups")
        return float(np.mean(values))

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
            y_true_matrix.append([parse_binary(row[schema["y_true"]], field=schema["y_true"], context=group_id) for row in group])
            y_pred_matrix.append([parse_binary(row[schema["y_pred"]], field=schema["y_pred"], context=group_id) for row in group])
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
            score_matrix_path = resolve_run_artifact(metric_inputs["score_matrix_csv"], run_dir=run_dir, field="metric_inputs.score_matrix_csv")
            recomputed = recompute_exact_metric_from_matrix(score_matrix_path, metric_name=metric_name, manifest=manifest, checks=checks)
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
    if not np.isfinite(recomputed):
        add_warn_or_fail(
            checks,
            gate=gate,
            fail_gate=STRICT_GATES,
            rule_id="PRIMARY_METRIC_RECOMPUTE",
            message=f"primary metric recomputed to a non-finite value: {recomputed}",
            fix="Regenerate prediction or metric_inputs artifacts so the primary metric is finite.",
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

    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fields = set(reader.fieldnames or [])
    except Exception as exc:
        add_check(
            checks,
            rule_id="PREDICTION_CSV_PARSE",
            severity="ERROR",
            status="FAIL",
            message=f"prediction CSV could not be parsed: {exc}",
            fix="Fix the prediction CSV encoding or structure.",
            decision_if_fail="DIAGNOSTIC_ONLY",
        )
        return

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
    metric_name = route_primary_metric(route_data)
    if schema["score"] is None:
        if route_uses_top4(route_data):
            add_check(
                checks,
                rule_id="PREDICTION_SCORE_COLUMN",
                severity="WARN",
                status="WARN",
                message="prediction CSV has no standard score column",
                fix="Use one of: score, y_score, probability, logit.",
            )
        else:
            add_check(checks, rule_id="PREDICTION_SCORE_COLUMN", severity="INFO", status="PASS", message="score column is not required for this no-Top4 route")
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

    if not top4_required:
        add_check(
            checks,
            rule_id="PREDICTION_TOP4_GROUPS",
            severity="INFO",
            status="PASS",
            message="Top-4 audit is not required for this route",
        )
    elif schema["subject_id"] and schema["trial_id"] and schema["pred_top4"]:
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
            n_pred = 0
            for row in group:
                try:
                    n_pred += parse_binary(row[schema["pred_top4"]], field=schema["pred_top4"], context=group_id)
                except ValueError:
                    n_pred = -1
                    break
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
        check_top4_group_semantics(checks, groups=groups, schema=schema, gate=gate)
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
        try:
            y_true = []
            y_pred = []
            for row in rows:
                y_true.append(parse_binary(row[schema["y_true"]], field=schema["y_true"], context="BA recompute"))
                y_pred.append(parse_binary(row[schema["y_pred"]], field=schema["y_pred"], context="BA recompute"))
            score = balanced_accuracy(np.array(y_true), np.array(y_pred))
        except ValueError as exc:
            add_warn_or_fail(
                checks,
                gate=gate,
                fail_gate=STRICT_GATES,
                rule_id="METRIC_RECOMPUTE_BA",
                message=f"BA recompute failed: {exc}",
                fix="Use binary y_true/y_pred values for labeled validation predictions.",
                decision_if_fail="DIAGNOSTIC_ONLY",
            )
        else:
            add_check(checks, rule_id="METRIC_RECOMPUTE_BA", severity="INFO", status="PASS", message=f"recomputed BA: {score:.6f}")
    elif metric_name == "no_top4_BA":
        add_warn_or_fail(
            checks,
            gate=gate,
            fail_gate=STRICT_GATES,
            rule_id="METRIC_RECOMPUTE_BA",
            message="y_true/y_pred columns are not present; BA recompute skipped",
            fix="For no_top4_BA validation runs, include y_true and y_pred in the prediction CSV.",
            decision_if_fail="DIAGNOSTIC_ONLY",
        )
    else:
        add_check(
            checks,
            rule_id="METRIC_RECOMPUTE_BA",
            severity="INFO",
            status="PASS",
            message=f"BA recompute skipped because {metric_name or 'the primary metric'} is checked by primary metric recompute",
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


def check_promotion_audit(route_id: str, promotion_path: Path, checks: list[AuditCheck], *, top4_required: bool = False) -> None:
    if not promotion_path.exists():
        add_check(
            checks,
            rule_id="PROMOTION_AUDIT_EXISTS",
            severity="ERROR",
            status="FAIL",
            message=f"promotion audit is missing: {promotion_path.relative_to(ROOT)}",
            fix="Write a promotion audit before using the promoted gate.",
            decision_if_fail="BLOCKED",
        )
        return

    add_check(checks, rule_id="PROMOTION_AUDIT_EXISTS", severity="INFO", status="PASS", message=f"promotion audit found: {promotion_path.relative_to(ROOT)}")
    fields = parse_key_value_markdown(promotion_path)
    missing = sorted(field for field in PROMOTION_REQUIRED_FIELDS if not fields.get(field))
    if missing:
        add_check(
            checks,
            rule_id="PROMOTION_AUDIT_FORMAT",
            severity="ERROR",
            status="FAIL",
            message="promotion audit missing required fields: " + ", ".join(missing),
            fix="Fill the promotion audit template with all required fields.",
            decision_if_fail="BLOCKED",
        )
        return
    if fields.get("route_id") != route_id:
        add_check(
            checks,
            rule_id="PROMOTION_AUDIT_FORMAT",
            severity="ERROR",
            status="FAIL",
            message="promotion audit route_id does not match audited route",
            fix="Use the promotion audit file for this route only.",
            decision_if_fail="BLOCKED",
        )
        return

    report_path = (ROOT / fields["candidate_audit_report"]).resolve()
    try:
        report_path.relative_to(ROOT.resolve())
    except ValueError:
        add_check(
            checks,
            rule_id="PROMOTION_AUDIT_CANDIDATE_REPORT",
            severity="ERROR",
            status="FAIL",
            message="candidate_audit_report escapes repository",
            fix="Point candidate_audit_report to an audit_report.json under the repository workspace.",
            decision_if_fail="BLOCKED",
        )
        return
    if not report_path.exists():
        add_check(
            checks,
            rule_id="PROMOTION_AUDIT_CANDIDATE_REPORT",
            severity="ERROR",
            status="FAIL",
            message="candidate_audit_report does not exist",
            fix="Reference the candidate gate audit_report.json for the promoted route.",
            decision_if_fail="BLOCKED",
        )
        return
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        add_check(
            checks,
            rule_id="PROMOTION_AUDIT_CANDIDATE_REPORT",
            severity="ERROR",
            status="FAIL",
            message=f"candidate_audit_report is not valid JSON: {exc}",
            fix="Reference a valid experiment audit JSON report.",
            decision_if_fail="BLOCKED",
        )
        return
    if report.get("route_id") != route_id or report.get("gate") != "candidate" or report.get("overall") != "PASS":
        add_check(
            checks,
            rule_id="PROMOTION_AUDIT_CANDIDATE_REPORT",
            severity="ERROR",
            status="FAIL",
            message="candidate_audit_report must be PASS for the same route under candidate gate",
            fix="Run and pass candidate gate before promoted review.",
            decision_if_fail="BLOCKED",
        )
    else:
        add_check(checks, rule_id="PROMOTION_AUDIT_CANDIDATE_REPORT", severity="INFO", status="PASS", message="promotion audit references a passing candidate audit report")

    report_checks = report.get("checks")
    if not isinstance(report_checks, list):
        add_check(
            checks,
            rule_id="PROMOTION_AUDIT_CANDIDATE_RULES",
            severity="ERROR",
            status="FAIL",
            message="candidate_audit_report has no checks list",
            fix="Reference a full experiment audit report, not a minimal status stub.",
            decision_if_fail="BLOCKED",
        )
        return
    statuses = {str(item.get("rule_id")): item.get("status") for item in report_checks if isinstance(item, dict)}
    required = set(PROMOTION_REQUIRED_CANDIDATE_RULES)
    if top4_required:
        required.update(PROMOTION_REQUIRED_TOP4_RULES)
    missing_or_failed = sorted(rule for rule in required if statuses.get(rule) != "PASS")
    if missing_or_failed:
        add_check(
            checks,
            rule_id="PROMOTION_AUDIT_CANDIDATE_RULES",
            severity="ERROR",
            status="FAIL",
            message="candidate audit report is missing passing critical rules: " + ", ".join(missing_or_failed),
            fix="Use a full passing candidate audit report with all critical evidence checks.",
            decision_if_fail="BLOCKED",
        )
    else:
        add_check(checks, rule_id="PROMOTION_AUDIT_CANDIDATE_RULES", severity="INFO", status="PASS", message="candidate audit report contains passing critical evidence rules")


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
                check_reproducibility_manifest(manifest, checks, gate=gate, root=ROOT)
                check_run_dataset_split_evidence(manifest, route_data, run_dir, checks, gate=gate)
                prediction_csv = manifest.get("prediction_csv")
                if isinstance(prediction_csv, str) and prediction_csv:
                    check_prediction_csv(resolve_run_artifact(prediction_csv, run_dir=run_dir, field="prediction_csv"), checks, route_data=route_data, gate=gate, manifest=manifest)
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

            if gate == "promoted":
                promotion_path = ROOT / "reports" / "promotion_audits" / f"{route_id}_promotion.md"
                check_promotion_audit(route_id, promotion_path, checks, top4_required=route_uses_top4(route_data))

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
