"""Posthoc score-matrix fusion and DEP/HC soft-router search utilities."""

from __future__ import annotations

import csv
import itertools
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from hust_bci_er.evaluation.exact_single_crop import assignment_grid, top4_predictions


KEY_COLUMNS = ("seed", "fold", "subject_id", "trial_id")
GROUP_COLUMNS = ("seed", "fold", "subject_id")
PROTOCOL_JOB_KEY = "protocol_job_key"
CANDIDATE_RESULT_COLUMNS = (
    "kind",
    "label",
    "mean_ba",
    "dep_mean_ba",
    "hc_mean_ba",
    "cohort_gap",
    "min_group_ba",
    "n_groups",
    "transform",
    "weights",
    "indices",
)
MOE_RESULT_COLUMNS = (
    "kind",
    "label",
    "mean_ba",
    "dep_mean_ba",
    "hc_mean_ba",
    "cohort_gap",
    "min_group_ba",
    "n_groups",
    "dep_idx",
    "hc_idx",
    "dep_label",
    "hc_label",
    "router_dep_recall",
    "router_hc_recall",
)


@dataclass(frozen=True)
class ScoreRun:
    label: str
    source: str
    keys: tuple[tuple[str, ...], ...]
    y_true: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class AlignedRuns:
    labels: tuple[str, ...]
    sources: tuple[str, ...]
    keys: tuple[tuple[str, ...], ...]
    y_true: np.ndarray
    scores: tuple[np.ndarray, ...]
    groups: tuple[np.ndarray, ...]
    group_subjects: tuple[str, ...]
    group_cohorts: np.ndarray


@dataclass(frozen=True)
class Candidate:
    kind: str
    label: str
    indices: tuple[int, ...]
    weights: tuple[float, ...]
    transform: str


_WORK_ALIGNED: AlignedRuns | None = None
_WORK_ASSIGNMENTS: np.ndarray | None = None
_WORK_TRANSFORM_CACHE: dict[tuple[int, str], np.ndarray] = {}
_WORK_ROUTER_DEP_RECALL: float = 0.65
_WORK_ROUTER_HC_RECALL: float = 0.75
_WORK_DEP_ROW_MASK: np.ndarray | None = None


def discover_assembled_runs(patterns: Sequence[str], *, root: Path) -> list[tuple[str, list[Path]]]:
    runs: list[tuple[str, list[Path]]] = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.name != "score_matrix.csv":
                continue
            label = "__".join(path.parts[-4:-1]) if len(path.parts) >= 4 else path.parent.name
            runs.append((f"assembled:{label}", [path]))
    return runs


def discover_expert_runs(
    summary_csv: Path,
    *,
    output_root: Path,
    top_n_each: int = 12,
) -> list[tuple[str, list[Path]]]:
    if not summary_csv.exists():
        return []
    df = pd.read_csv(summary_csv)
    selected: list[str] = []
    for metric in ("p1_mean", "dep_mean", "hc_mean"):
        if metric in df.columns:
            selected.extend(df.sort_values(metric, ascending=False)["route"].head(top_n_each).astype(str).tolist())
    routes = list(dict.fromkeys(selected))
    runs: list[tuple[str, list[Path]]] = []
    for route in routes:
        matches = sorted(output_root.glob(f"expert_top4_*_{route}/{route}/job_runs/*/score_matrix.csv"))
        if not matches:
            matches = sorted(output_root.glob(f"expert_top4_*{route}*/{route}/job_runs/*/score_matrix.csv"))
        if matches:
            runs.append((f"expert:{route}", matches))
    return runs


def load_score_run(label: str, paths: Sequence[Path]) -> ScoreRun:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        crop_cols = _crop_columns(df.columns)
        missing = [col for col in ("subject_id", "trial_id", "y_true") if col not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        inferred = _infer_job_metadata(path)
        for col in ("protocol_job", "seed", "fold"):
            if col not in df.columns:
                df[col] = inferred.get(col, "")
            else:
                values = df[col].astype(str)
                if values.replace({"nan": "", "None": ""}).str.strip().eq("").all():
                    df[col] = inferred.get(col, "")
        df[PROTOCOL_JOB_KEY] = df["protocol_job"].map(_normalize_protocol_job)
        frames.append(
            df[[PROTOCOL_JOB_KEY, *KEY_COLUMNS, "y_true", *crop_cols]].rename(
                columns={crop_cols[idx]: f"crop_{idx}" for idx in range(len(crop_cols))}
            )
        )
    if not frames:
        raise ValueError(f"no score matrices for {label}")
    merged = pd.concat(frames, ignore_index=True)
    for col in (PROTOCOL_JOB_KEY, *KEY_COLUMNS):
        merged[col] = merged[col].astype(str)
    key_columns = list(KEY_COLUMNS)
    if merged.duplicated(key_columns).any():
        key_columns = [PROTOCOL_JOB_KEY, *KEY_COLUMNS]
    merged = merged.sort_values(key_columns, kind="mergesort")
    if merged.duplicated(key_columns).any():
        dup = merged[merged.duplicated(key_columns, keep=False)].head(3)
        raise ValueError(f"duplicate score keys for {label}: {dup[key_columns].to_dict(orient='records')}")
    scores = merged[[f"crop_{idx}" for idx in range(5)]].to_numpy(dtype=np.float64)
    y_true = merged["y_true"].to_numpy(dtype=np.int8)
    keys = tuple(tuple(row) for row in merged[key_columns].to_numpy(dtype=str))
    return ScoreRun(label=label, source=";".join(str(p) for p in paths[:3]), keys=keys, y_true=y_true, scores=scores)


def _infer_job_metadata(path: Path) -> dict[str, str]:
    job_name = path.parent.name
    seed = ""
    fold = ""
    seed_match = re.search(r"seed(\d+)", job_name)
    fold_match = re.search(r"fold(\d+)", job_name)
    if seed_match:
        seed = seed_match.group(1)
    if fold_match:
        fold = fold_match.group(1)
    if "job_runs" not in {part for part in path.parts}:
        return {"protocol_job": "", "seed": seed, "fold": fold}
    return {"protocol_job": job_name, "seed": seed, "fold": fold}


def _normalize_protocol_job(protocol_job: object) -> str:
    job = str(protocol_job or "").strip()
    if not job:
        return ""
    protocol_match = re.search(r"\b(p[123])\b", job)
    crop_match = re.search(r"(?:^|[_-])(crop\d+|fixed_crop\d+|eval_crop\d+)(?:$|[_-])", job)
    seed_match = re.search(r"seed(\d+)", job)
    fold_match = re.search(r"fold(\d+)", job)
    parts = []
    if protocol_match:
        parts.append(protocol_match.group(1))
    if crop_match:
        parts.append(crop_match.group(1))
    if seed_match:
        parts.append(f"seed{seed_match.group(1)}")
    if fold_match:
        parts.append(f"fold{fold_match.group(1)}")
    return "__".join(parts) if parts else job


def align_score_runs(runs: Sequence[ScoreRun]) -> AlignedRuns:
    if len(runs) < 2:
        raise ValueError("need at least two score runs for fusion search")
    common = set(runs[0].keys)
    for run in runs[1:]:
        common &= set(run.keys)
    if not common:
        raise ValueError("score runs have no common alignment keys")
    common_keys = tuple(sorted(common))
    aligned_scores: list[np.ndarray] = []
    y_ref: np.ndarray | None = None
    for run in runs:
        index = {key: idx for idx, key in enumerate(run.keys)}
        order = np.array([index[key] for key in common_keys], dtype=np.int64)
        y = run.y_true[order]
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y_ref, y):
            raise ValueError(f"y_true mismatch after alignment for {run.label}")
        aligned_scores.append(run.scores[order])
    groups, subjects, cohorts = _build_groups(common_keys, y_ref if y_ref is not None else np.empty(0, dtype=np.int8))
    return AlignedRuns(
        labels=tuple(run.label for run in runs),
        sources=tuple(run.source for run in runs),
        keys=common_keys,
        y_true=y_ref if y_ref is not None else np.empty(0, dtype=np.int8),
        scores=tuple(aligned_scores),
        groups=groups,
        group_subjects=subjects,
        group_cohorts=cohorts,
    )


def run_posthoc_search(
    aligned: AlignedRuns,
    *,
    workers: int = 12,
    max_base_runs: int = 24,
    pair_weight_step: float = 0.05,
    triple_weight_step: float = 0.10,
    screen_assignments: int = 4096,
    exact_top_k: int = 80,
    fusion_asha_candidates: int = 4096,
    fusion_asha_max_width: int = 4,
    fusion_asha_eta: int = 3,
    fusion_asha_min_groups: int = 48,
    router_dep_recall: float = 0.65,
    router_hc_recall: float = 0.75,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(20260525)
    screen_grid = rng.integers(0, 5, size=(int(screen_assignments), 8), dtype=np.int16)
    exact_grid = assignment_grid(8, 5)

    single_rows = []
    for idx, label in enumerate(aligned.labels):
        row = _metric_row(aligned.scores[idx], aligned, screen_grid, candidate_label=label, kind="single_screen")
        row["run_idx"] = idx
        single_rows.append(row)
    single_df = pd.DataFrame(single_rows).sort_values("mean_ba", ascending=False)

    selected = _selected_indices(single_df, max_base_runs=max_base_runs)
    candidates = _fusion_candidates(selected, aligned.labels, pair_weight_step=pair_weight_step, triple_weight_step=triple_weight_step)
    fusion_screen = _evaluate_candidates(candidates, aligned, screen_grid, workers=workers)
    top_candidates = _top_candidates(fusion_screen, candidates, exact_top_k)
    fusion_exact = _evaluate_candidates(top_candidates, aligned, exact_grid, workers=workers)

    fusion_asha_screen, fusion_asha_exact = _run_fusion_asha_search(
        selected,
        aligned,
        screen_grid,
        exact_grid,
        workers=workers,
        candidate_count=fusion_asha_candidates,
        max_width=fusion_asha_max_width,
        eta=fusion_asha_eta,
        min_groups=fusion_asha_min_groups,
        exact_top_k=exact_top_k,
        rng=rng,
    )

    moe_candidates = _moe_candidates(single_df, max_base_runs=max_base_runs)
    moe_screen = _evaluate_moe_candidates(
        moe_candidates,
        aligned,
        screen_grid,
        workers=workers,
        router_dep_recall=router_dep_recall,
        router_hc_recall=router_hc_recall,
    )
    top_moe = _top_moe_candidates(moe_screen, moe_candidates, exact_top_k)
    moe_exact = _evaluate_moe_candidates(
        top_moe,
        aligned,
        exact_grid,
        workers=workers,
        router_dep_recall=router_dep_recall,
        router_hc_recall=router_hc_recall,
    )

    return {
        "single_screen": single_df,
        "fusion_screen": fusion_screen,
        "fusion_exact": fusion_exact,
        "fusion_asha_screen": fusion_asha_screen,
        "fusion_asha_exact": fusion_asha_exact,
        "moe_screen": moe_screen,
        "moe_exact": moe_exact,
    }


def write_search_outputs(results: dict[str, pd.DataFrame], out_dir: Path, *, config: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    report = render_report(results, config=config)
    (out_dir / "fusion_moe_search_report.md").write_text(report, encoding="utf-8")


def render_report(results: dict[str, pd.DataFrame], *, config: dict[str, object]) -> str:
    lines = [
        "# Fusion/MoE Posthoc Search",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, indent=2, sort_keys=True),
        "```",
        "",
    ]
    for name in ("single_screen", "fusion_exact", "fusion_asha_exact", "moe_exact"):
        df = results[name].head(20)
        lines.extend([f"## {name}", "", _markdown_table(df), ""])
    return "\n".join(lines)


def _crop_columns(columns: Iterable[str]) -> list[str]:
    cols = list(columns)
    underscored = [f"crop_{idx}" for idx in range(5)]
    compact = [f"crop{idx}" for idx in range(5)]
    if all(col in cols for col in underscored):
        return underscored
    if all(col in cols for col in compact):
        return compact
    raise ValueError("score_matrix must contain crop_0..crop_4 or crop0..crop4")


def _key_frame(keys: tuple[tuple[str, ...], ...]) -> pd.DataFrame:
    if not keys:
        return pd.DataFrame(columns=KEY_COLUMNS)
    key_width = len(keys[0])
    if key_width == len(KEY_COLUMNS):
        return pd.DataFrame(keys, columns=KEY_COLUMNS)
    if key_width == len(KEY_COLUMNS) + 1:
        return pd.DataFrame(keys, columns=(PROTOCOL_JOB_KEY, *KEY_COLUMNS))
    raise ValueError(f"unsupported score key width: {key_width}")


def _build_groups(keys: tuple[tuple[str, ...], ...], y_true: np.ndarray) -> tuple[tuple[np.ndarray, ...], tuple[str, ...], np.ndarray]:
    key_df = _key_frame(keys)
    group_columns = list(GROUP_COLUMNS)
    if PROTOCOL_JOB_KEY in key_df.columns:
        group_columns = [PROTOCOL_JOB_KEY, *group_columns]
    groups: list[np.ndarray] = []
    subjects: list[str] = []
    cohorts: list[int] = []
    for _, idx in key_df.groupby(group_columns, sort=True).indices.items():
        arr = np.asarray(sorted(idx), dtype=np.int64)
        if arr.size != 8:
            continue
        truth = y_true[arr]
        if truth.sum() != 4:
            continue
        subject = str(key_df.iloc[arr[0]]["subject_id"])
        groups.append(arr)
        subjects.append(subject)
        cohorts.append(1 if subject.upper().startswith("DEP") else 0)
    if not groups:
        raise ValueError("no valid 8-trial, 4-positive groups found")
    return tuple(groups), tuple(subjects), np.asarray(cohorts, dtype=np.int8)


def _metric_row(scores: np.ndarray, aligned: AlignedRuns, assignments: np.ndarray, *, candidate_label: str, kind: str) -> dict[str, object]:
    group_scores = []
    dep_scores = []
    hc_scores = []
    min_group = 1.0
    for group_idx, idx in enumerate(aligned.groups):
        mat = scores[idx]
        y = aligned.y_true[idx]
        ba = _assignment_ba(mat, y, assignments)
        group_scores.append(ba)
        min_group = min(min_group, ba)
        if int(aligned.group_cohorts[group_idx]) == 1:
            dep_scores.append(ba)
        else:
            hc_scores.append(ba)
    dep_mean = float(np.mean(dep_scores)) if dep_scores else float("nan")
    hc_mean = float(np.mean(hc_scores)) if hc_scores else float("nan")
    return {
        "kind": kind,
        "label": candidate_label,
        "mean_ba": float(np.mean(group_scores)),
        "dep_mean_ba": dep_mean,
        "hc_mean_ba": hc_mean,
        "cohort_gap": float(abs(dep_mean - hc_mean)) if math.isfinite(dep_mean) and math.isfinite(hc_mean) else float("nan"),
        "min_group_ba": float(min_group),
        "n_groups": len(group_scores),
    }


def _assignment_ba(mat: np.ndarray, y_true: np.ndarray, assignments: np.ndarray) -> float:
    selected = mat[np.arange(mat.shape[0]), assignments]
    pred = top4_predictions(selected)
    return float((pred == y_true[None, :]).mean(axis=1).mean())


def _selected_indices(single_df: pd.DataFrame, *, max_base_runs: int) -> tuple[int, ...]:
    picks: list[int] = []
    for col in ("mean_ba", "dep_mean_ba", "hc_mean_ba"):
        picks.extend(single_df.sort_values(col, ascending=False)["run_idx"].head(max_base_runs // 2).astype(int).tolist())
    picks.extend(single_df["run_idx"].head(max_base_runs).astype(int).tolist())
    return tuple(list(dict.fromkeys(picks))[:max_base_runs])


def _run_fusion_asha_search(
    selected: Sequence[int],
    aligned: AlignedRuns,
    screen_grid: np.ndarray,
    exact_grid: np.ndarray,
    *,
    workers: int,
    candidate_count: int,
    max_width: int,
    eta: int,
    min_groups: int,
    exact_top_k: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = _fusion_asha_candidates(
        selected,
        aligned.labels,
        candidate_count=max(0, int(candidate_count)),
        max_width=max(2, int(max_width)),
        rng=rng,
    )
    if not candidates:
        empty = pd.DataFrame()
        return empty, empty

    group_count = len(aligned.groups)
    budgets = _asha_group_budgets(group_count, min_groups=max(1, int(min_groups)), eta=max(2, int(eta)))
    group_order = rng.permutation(group_count)
    active = candidates
    history = []
    for stage, budget in enumerate(budgets):
        budget_groups = np.sort(group_order[:budget])
        subset = _with_group_subset(aligned, budget_groups)
        stage_df = _evaluate_candidates(active, subset, screen_grid, workers=workers)
        stage_df["asha_stage"] = stage
        stage_df["asha_budget_groups"] = int(budget)
        stage_df["asha_active_candidates"] = len(active)
        history.append(stage_df)
        keep = max(int(exact_top_k), int(math.ceil(len(active) / max(2, int(eta)))))
        active = _top_candidates(stage_df, active, min(keep, len(active)))
        if len(active) <= exact_top_k:
            break

    screen_df = pd.concat(history, ignore_index=True) if history else pd.DataFrame()
    exact_df = _evaluate_candidates(active[:exact_top_k], aligned, exact_grid, workers=workers)
    exact_df["asha_exact_from_candidates"] = len(active)
    return screen_df.sort_values(["asha_stage", "mean_ba", "min_group_ba"], ascending=[False, False, False]), exact_df


def _fusion_asha_candidates(
    selected: Sequence[int],
    labels: Sequence[str],
    *,
    candidate_count: int,
    max_width: int,
    rng: np.random.Generator,
) -> list[Candidate]:
    selected = tuple(dict.fromkeys(int(idx) for idx in selected))
    if len(selected) < 2 or candidate_count <= 0:
        return []
    transforms = ("raw", "global_z", "sigmoid_t0p5", "sigmoid_t0p75", "sigmoid_t1p0", "sigmoid_t1p25", "sigmoid_t1p5")
    max_width = min(max_width, len(selected))
    candidates: dict[str, Candidate] = {}

    def add(candidate: Candidate) -> None:
        candidates.setdefault(candidate.label, candidate)

    for idx in selected:
        for transform in ("raw", "global_z", "sigmoid_t0p75", "sigmoid_t1p25"):
            add(Candidate("asha_single", f"asha:{transform}:single:{labels[idx]}", (idx,), (1.0,), transform))

    for i, j in itertools.combinations(selected[: min(10, len(selected))], 2):
        for weight in (0.25, 0.35, 0.5, 0.65, 0.75):
            for transform in ("raw", "global_z"):
                label = f"asha:{transform}:pair:{labels[i]}:{weight:.2f}+{labels[j]}:{1.0 - weight:.2f}"
                add(Candidate("asha_weighted", label, (i, j), (weight, 1.0 - weight), transform))

    attempts = 0
    max_attempts = max(1000, candidate_count * 20)
    while len(candidates) < candidate_count and attempts < max_attempts:
        attempts += 1
        width = int(rng.integers(2, max_width + 1))
        indices = tuple(int(idx) for idx in rng.choice(selected, size=width, replace=False))
        alpha = float(rng.choice(np.asarray([0.35, 0.6, 1.0, 1.8], dtype=np.float64)))
        weights = tuple(float(x) for x in rng.dirichlet(np.full(width, alpha)))
        transform = str(rng.choice(np.asarray(transforms)))
        label = "asha:{transform}:w{width}:".format(transform=transform, width=width) + "+".join(
            f"{labels[idx]}:{weight:.3f}" for idx, weight in zip(indices, weights)
        )
        add(Candidate("asha_weighted", label, indices, weights, transform))
    return list(candidates.values())[:candidate_count]


def _asha_group_budgets(group_count: int, *, min_groups: int, eta: int) -> tuple[int, ...]:
    if group_count <= min_groups:
        return (group_count,)
    budgets = [min(min_groups, group_count)]
    while budgets[-1] < group_count:
        next_budget = min(group_count, max(budgets[-1] + 1, budgets[-1] * eta))
        budgets.append(next_budget)
    return tuple(dict.fromkeys(budgets))


def _with_group_subset(aligned: AlignedRuns, group_indices: np.ndarray) -> AlignedRuns:
    picks = [int(idx) for idx in group_indices]
    return AlignedRuns(
        labels=aligned.labels,
        sources=aligned.sources,
        keys=aligned.keys,
        y_true=aligned.y_true,
        scores=aligned.scores,
        groups=tuple(aligned.groups[idx] for idx in picks),
        group_subjects=tuple(aligned.group_subjects[idx] for idx in picks),
        group_cohorts=aligned.group_cohorts[picks],
    )


def _fusion_candidates(
    selected: Sequence[int],
    labels: Sequence[str],
    *,
    pair_weight_step: float,
    triple_weight_step: float,
) -> list[Candidate]:
    transforms = ("raw", "global_z", "sigmoid_t0p75", "sigmoid_t1p25")
    candidates: list[Candidate] = []
    for idx in selected:
        candidates.append(Candidate("single", str(labels[idx]), (idx,), (1.0,), "raw"))
    for i, j in itertools.combinations(selected, 2):
        weight = 0.0
        while weight <= 1.000001:
            w = round(float(weight), 6)
            for transform in transforms:
                label = f"{transform}:pair:{labels[i]}:{w:.2f}+{labels[j]}:{1.0 - w:.2f}"
                candidates.append(Candidate("weighted_pair", label, (i, j), (w, 1.0 - w), transform))
            weight += pair_weight_step
    top_for_triples = tuple(selected[: min(10, len(selected))])
    for combo in itertools.combinations(top_for_triples, 3):
        for w0 in _grid_values(triple_weight_step):
            for w1 in _grid_values(triple_weight_step):
                w2 = round(1.0 - w0 - w1, 6)
                if w2 < -1e-9:
                    continue
                weights = (round(w0, 6), round(w1, 6), round(w2, 6))
                if min(weights) < 0.0:
                    continue
                label = "global_z:triple:" + "+".join(f"{labels[idx]}:{weight:.2f}" for idx, weight in zip(combo, weights))
                candidates.append(Candidate("weighted_triple", label, combo, weights, "global_z"))
    return candidates


def _grid_values(step: float) -> tuple[float, ...]:
    values = []
    value = 0.0
    while value <= 1.000001:
        values.append(round(value, 6))
        value += step
    return tuple(values)


def _evaluate_candidates(candidates: Sequence[Candidate], aligned: AlignedRuns, assignments: np.ndarray, *, workers: int) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame(columns=CANDIDATE_RESULT_COLUMNS)
    if workers > 1:
        chunksize = max(1, len(candidates) // (int(workers) * 8))
        with ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=_init_candidate_worker,
            initargs=(aligned, assignments),
        ) as pool:
            rows = list(pool.map(_evaluate_candidate_worker, candidates, chunksize=chunksize))
        return pd.DataFrame(rows).sort_values(["mean_ba", "min_group_ba"], ascending=False)

    transformed_cache: dict[tuple[int, str], np.ndarray] = {}

    def transformed(idx: int, mode: str) -> np.ndarray:
        key = (idx, mode)
        if key not in transformed_cache:
            transformed_cache[key] = _transform(aligned.scores[idx], mode)
        return transformed_cache[key]

    def evaluate(candidate: Candidate) -> dict[str, object]:
        fused = np.zeros_like(aligned.scores[candidate.indices[0]], dtype=np.float64)
        for idx, weight in zip(candidate.indices, candidate.weights):
            fused += float(weight) * transformed(idx, candidate.transform)
        row = _metric_row(fused, aligned, assignments, candidate_label=candidate.label, kind=candidate.kind)
        row["transform"] = candidate.transform
        row["weights"] = ",".join(f"{w:.6g}" for w in candidate.weights)
        row["indices"] = ",".join(str(i) for i in candidate.indices)
        return row

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        rows = list(pool.map(evaluate, candidates))
    return pd.DataFrame(rows).sort_values(["mean_ba", "min_group_ba"], ascending=False)


def _init_candidate_worker(aligned: AlignedRuns, assignments: np.ndarray) -> None:
    global _WORK_ALIGNED, _WORK_ASSIGNMENTS, _WORK_TRANSFORM_CACHE
    _WORK_ALIGNED = aligned
    _WORK_ASSIGNMENTS = assignments
    _WORK_TRANSFORM_CACHE = {}


def _evaluate_candidate_worker(candidate: Candidate) -> dict[str, object]:
    if _WORK_ALIGNED is None or _WORK_ASSIGNMENTS is None:
        raise RuntimeError("candidate worker is not initialized")
    aligned = _WORK_ALIGNED

    def transformed(idx: int, mode: str) -> np.ndarray:
        key = (idx, mode)
        if key not in _WORK_TRANSFORM_CACHE:
            _WORK_TRANSFORM_CACHE[key] = _transform(aligned.scores[idx], mode)
        return _WORK_TRANSFORM_CACHE[key]

    fused = np.zeros_like(aligned.scores[candidate.indices[0]], dtype=np.float64)
    for idx, weight in zip(candidate.indices, candidate.weights):
        fused += float(weight) * transformed(idx, candidate.transform)
    row = _metric_row(fused, aligned, _WORK_ASSIGNMENTS, candidate_label=candidate.label, kind=candidate.kind)
    row["transform"] = candidate.transform
    row["weights"] = ",".join(f"{w:.6g}" for w in candidate.weights)
    row["indices"] = ",".join(str(i) for i in candidate.indices)
    return row


def _transform(scores: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if mode == "raw":
        return arr
    if mode == "global_z":
        std = float(arr.std())
        return (arr - float(arr.mean())) / (std if std > 1e-8 else 1.0)
    if mode.startswith("sigmoid_t"):
        temp = float(mode.removeprefix("sigmoid_t").replace("p", "."))
        return 1.0 / (1.0 + np.exp(-np.clip(arr / max(temp, 1e-6), -50.0, 50.0)))
    raise ValueError(f"unknown transform: {mode}")


def _top_candidates(df: pd.DataFrame, candidates: Sequence[Candidate], top_k: int) -> list[Candidate]:
    if df.empty or "label" not in df.columns or top_k <= 0:
        return []
    lookup = {candidate.label: candidate for candidate in candidates}
    return [lookup[str(label)] for label in df["label"].head(top_k) if str(label) in lookup]


def _moe_candidates(single_df: pd.DataFrame, *, max_base_runs: int) -> list[tuple[int, int, str]]:
    dep = single_df.sort_values(["dep_mean_ba", "mean_ba"], ascending=False)["run_idx"].head(max_base_runs).astype(int).tolist()
    hc = single_df.sort_values(["hc_mean_ba", "mean_ba"], ascending=False)["run_idx"].head(max_base_runs).astype(int).tolist()
    mean = single_df.sort_values("mean_ba", ascending=False)["run_idx"].head(max_base_runs // 2).astype(int).tolist()
    dep = list(dict.fromkeys(dep + mean))
    hc = list(dict.fromkeys(hc + mean))
    return [(d, h, "router_expected_soft") for d in dep for h in hc]


def _evaluate_moe_candidates(
    candidates: Sequence[tuple[int, int, str]],
    aligned: AlignedRuns,
    assignments: np.ndarray,
    *,
    workers: int,
    router_dep_recall: float,
    router_hc_recall: float,
) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame(columns=MOE_RESULT_COLUMNS)
    key_df = _key_frame(aligned.keys)
    dep_row_mask = key_df["subject_id"].astype(str).str.upper().str.startswith("DEP").to_numpy(dtype=bool)
    if workers > 1:
        chunksize = max(1, len(candidates) // (int(workers) * 8))
        with ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=_init_moe_worker,
            initargs=(aligned, assignments, dep_row_mask, router_dep_recall, router_hc_recall),
        ) as pool:
            rows = list(pool.map(_evaluate_moe_worker, candidates, chunksize=chunksize))
        return pd.DataFrame(rows).sort_values(["mean_ba", "min_group_ba"], ascending=False)

    def evaluate(item: tuple[int, int, str]) -> dict[str, object]:
        dep_idx, hc_idx, kind = item
        dep_scores = _transform(aligned.scores[dep_idx], "global_z")
        hc_scores = _transform(aligned.scores[hc_idx], "global_z")
        fused = np.where(
            dep_row_mask[:, None],
            router_dep_recall * dep_scores + (1.0 - router_dep_recall) * hc_scores,
            (1.0 - router_hc_recall) * dep_scores + router_hc_recall * hc_scores,
        )
        label = f"dep={aligned.labels[dep_idx]} | hc={aligned.labels[hc_idx]} | dep_recall={router_dep_recall:.2f} hc_recall={router_hc_recall:.2f}"
        row = _metric_row(fused, aligned, assignments, candidate_label=label, kind=kind)
        row["dep_idx"] = dep_idx
        row["hc_idx"] = hc_idx
        row["dep_label"] = aligned.labels[dep_idx]
        row["hc_label"] = aligned.labels[hc_idx]
        row["router_dep_recall"] = router_dep_recall
        row["router_hc_recall"] = router_hc_recall
        return row

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        rows = list(pool.map(evaluate, candidates))
    return pd.DataFrame(rows).sort_values(["mean_ba", "min_group_ba"], ascending=False)


def _init_moe_worker(
    aligned: AlignedRuns,
    assignments: np.ndarray,
    dep_row_mask: np.ndarray,
    router_dep_recall: float,
    router_hc_recall: float,
) -> None:
    global _WORK_ALIGNED, _WORK_ASSIGNMENTS, _WORK_TRANSFORM_CACHE, _WORK_DEP_ROW_MASK, _WORK_ROUTER_DEP_RECALL, _WORK_ROUTER_HC_RECALL
    _WORK_ALIGNED = aligned
    _WORK_ASSIGNMENTS = assignments
    _WORK_TRANSFORM_CACHE = {}
    _WORK_DEP_ROW_MASK = dep_row_mask
    _WORK_ROUTER_DEP_RECALL = float(router_dep_recall)
    _WORK_ROUTER_HC_RECALL = float(router_hc_recall)


def _evaluate_moe_worker(item: tuple[int, int, str]) -> dict[str, object]:
    if _WORK_ALIGNED is None or _WORK_ASSIGNMENTS is None or _WORK_DEP_ROW_MASK is None:
        raise RuntimeError("moe worker is not initialized")
    aligned = _WORK_ALIGNED
    dep_idx, hc_idx, kind = item
    dep_scores = _transform(aligned.scores[dep_idx], "global_z")
    hc_scores = _transform(aligned.scores[hc_idx], "global_z")
    fused = np.where(
        _WORK_DEP_ROW_MASK[:, None],
        _WORK_ROUTER_DEP_RECALL * dep_scores + (1.0 - _WORK_ROUTER_DEP_RECALL) * hc_scores,
        (1.0 - _WORK_ROUTER_HC_RECALL) * dep_scores + _WORK_ROUTER_HC_RECALL * hc_scores,
    )
    label = f"dep={aligned.labels[dep_idx]} | hc={aligned.labels[hc_idx]} | dep_recall={_WORK_ROUTER_DEP_RECALL:.2f} hc_recall={_WORK_ROUTER_HC_RECALL:.2f}"
    row = _metric_row(fused, aligned, _WORK_ASSIGNMENTS, candidate_label=label, kind=kind)
    row["dep_idx"] = dep_idx
    row["hc_idx"] = hc_idx
    row["dep_label"] = aligned.labels[dep_idx]
    row["hc_label"] = aligned.labels[hc_idx]
    row["router_dep_recall"] = _WORK_ROUTER_DEP_RECALL
    row["router_hc_recall"] = _WORK_ROUTER_HC_RECALL
    return row


def _top_moe_candidates(df: pd.DataFrame, candidates: Sequence[tuple[int, int, str]], top_k: int) -> list[tuple[int, int, str]]:
    if df.empty or top_k <= 0:
        return []
    wanted = {(int(row.dep_idx), int(row.hc_idx), str(row.kind)) for row in df.head(top_k).itertuples(index=False)}
    return [item for item in candidates if item in wanted]


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    keep = [
        col
        for col in (
            "kind",
            "label",
            "mean_ba",
            "dep_mean_ba",
            "hc_mean_ba",
            "cohort_gap",
            "min_group_ba",
            "weights",
            "transform",
            "dep_label",
            "hc_label",
        )
        if col in df.columns
    ]
    rows = df[keep].copy()
    for col in rows.columns:
        if pd.api.types.is_float_dtype(rows[col]):
            rows[col] = rows[col].map(lambda value: f"{float(value):.6f}")
    header = "| " + " | ".join(keep) + " |"
    sep = "| " + " | ".join("---" for _ in keep) + " |"
    body = []
    for row in rows.itertuples(index=False):
        body.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    return "\n".join([header, sep, *body])
