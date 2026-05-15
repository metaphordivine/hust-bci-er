"""Static scanner for leakage-prone source references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


BLOCKED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("public_label", re.compile(r"(?<![a-z0-9_])public_labels?(?![a-z0-9_])")),
    ("public labels", re.compile(r"(?<![a-z0-9])public labels?(?![a-z0-9])")),
    ("private_label", re.compile(r"(?<![a-z0-9_])private_labels?(?![a-z0-9_])")),
    ("private labels", re.compile(r"(?<![a-z0-9])private labels?(?![a-z0-9])")),
    ("leaderboard_feedback", re.compile(r"(?<![a-z0-9_])leaderboard_feedback(?![a-z0-9_])")),
    ("leaderboard feedback", re.compile(r"(?<![a-z0-9])leaderboard feedback(?![a-z0-9])")),
    ("public_y", re.compile(r"(?<![a-z0-9_])public_y(?![a-z0-9_])")),
    ("private_y", re.compile(r"(?<![a-z0-9_])private_y(?![a-z0-9_])")),
    ("public_truth", re.compile(r"(?<![a-z0-9_])public_truth(?![a-z0-9_])")),
    ("private_truth", re.compile(r"(?<![a-z0-9_])private_truth(?![a-z0-9_])")),
    ("pseudo_public_gt", re.compile(r"(?<![a-z0-9_])pseudo_public_gt(?![a-z0-9_])")),
)
SCAN_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".json"}
EXCLUDED_REL_PATHS = {"src/hust_bci_er/audit/source_scanner.py", "scripts/scan_no_leakage.py"}
RUNTIME_REL_ROOTS = (
    "src/hust_bci_er/features",
    "src/hust_bci_er/inference",
    "src/hust_bci_er/models",
    "src/hust_bci_er/preprocessing",
    "src/hust_bci_er/training",
    "scripts",
)
MODEL_FEATURE_REL_ROOTS = (
    "src/hust_bci_er/features",
    "src/hust_bci_er/models",
    "src/hust_bci_er/preprocessing",
    "src/hust_bci_er/training",
    "scripts",
)
INFERENCE_REL_ROOTS = ("src/hust_bci_er/inference",)
INFERENCE_SCRIPT_HINTS = ("infer", "inference", "predict", "prediction", "submit", "assemble_score_route")
INFERENCE_LABEL_AUDIT_ALLOWLIST = {"src/hust_bci_er/inference/score_route_assembly.py"}
ID_SHORTCUT_KEYS = ("subject_id", "trial_id", "pseudo_trial_id", "file_name", "filename")
ID_AS_FEATURE_PATTERNS = (
    re.compile(r"\b(feature|features|input|inputs|x)\b.*\b(subject_id|trial_id|pseudo_trial_id|file_name|filename)\b"),
    re.compile(r"\b(subject_id|trial_id|pseudo_trial_id|file_name|filename)\b.*\b(feature|features|input|inputs|x)\b"),
    re.compile(r"\b(hash|one_hot|labelencoder|factorize|get_dummies|encode|embedding)\b.*\b(subject_id|trial_id|pseudo_trial_id|file_name|filename)\b"),
)
MODEL_LABEL_PATTERNS = (
    re.compile(r"(?<![a-z0-9_])(y_true|ground_truth|label|labels)(?![a-z0-9_])"),
)
INDIRECT_LABEL_FEATURE_PATTERNS = (
    re.compile(r"\b(features?|inputs?|x)\b.*\b(label_path|labels_path|target_path|ground_truth_path)\b"),
    re.compile(r"\b(label_path|labels_path|target_path|ground_truth_path)\b.*\b(features?|inputs?|x)\b"),
    re.compile(r"\bmerge\s*\([^)]*(y_true|label|labels|ground_truth)"),
    re.compile(r"\bjoin\s*\([^)]*(y_true|label|labels|ground_truth)"),
)


@dataclass(frozen=True)
class SourceFinding:
    path: str
    pattern: str
    line: int


def iter_scan_files(root: Path, rel_roots: tuple[str, ...] = ("src", "scripts", "configs")):
    root = root.resolve()
    for rel in rel_roots:
        base = root / rel
        if not base.exists():
            continue
        for path in base.rglob("*"):
            rel_path = path.relative_to(root).as_posix()
            if path.is_file() and path.suffix.lower() in SCAN_SUFFIXES and rel_path not in EXCLUDED_REL_PATHS:
                yield path


def is_under_any(path: Path, root: Path, rel_roots: tuple[str, ...]) -> bool:
    resolved = path.resolve()
    for rel in rel_roots:
        try:
            resolved.relative_to((root / rel).resolve())
            return True
        except ValueError:
            continue
    return False


def is_inference_script(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        return False
    if not rel.startswith("scripts/"):
        return False
    name = path.stem.lower()
    return any(hint in name for hint in INFERENCE_SCRIPT_HINTS)


def allows_inference_label_reference(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        return False
    return rel in INFERENCE_LABEL_AUDIT_ALLOWLIST


def scan_no_leakage(root: Path) -> list[SourceFinding]:
    findings: list[SourceFinding] = []
    for path in iter_scan_files(root):
        in_runtime = is_under_any(path, root, RUNTIME_REL_ROOTS)
        in_model_feature_path = is_under_any(path, root, MODEL_FEATURE_REL_ROOTS)
        in_inference = is_under_any(path, root, INFERENCE_REL_ROOTS) or is_inference_script(path, root)
        for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            lowered = line.lower()
            for pattern, regex in BLOCKED_PATTERNS:
                if regex.search(lowered):
                    findings.append(SourceFinding(path.relative_to(root).as_posix(), pattern, idx))
            if in_runtime:
                for key in ID_SHORTCUT_KEYS:
                    if f"model_feature:allow_{key}" in lowered:
                        break
                else:
                    if in_model_feature_path and any(regex.search(lowered) for regex in ID_AS_FEATURE_PATTERNS):
                        findings.append(SourceFinding(path.relative_to(root).as_posix(), "id_shortcut_as_feature", idx))
                    if in_model_feature_path and any(regex.search(lowered) for regex in INDIRECT_LABEL_FEATURE_PATTERNS):
                        findings.append(SourceFinding(path.relative_to(root).as_posix(), "indirect_label_feature", idx))
                    if in_inference and not allows_inference_label_reference(path, root) and any(regex.search(lowered) for regex in MODEL_LABEL_PATTERNS):
                        findings.append(SourceFinding(path.relative_to(root).as_posix(), "inference_label_reference", idx))
    return findings
