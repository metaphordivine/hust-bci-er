"""Batch-execute P1 candidate runs for torch_classifier routes that passed diagnostic.

Reads a batch diagnostic summary JSON (produced by
``run_all_torch_routes_diagnostic.py``) and re-runs the passed routes
with candidate gate (no epoch override, full 5-fold, 5 seeds).

Usage::

    python scripts/run_all_torch_routes_candidate.py \
        --from-diagnostic outputs/batch_diagnostic/batch_summary.json \
        --data-root <path> \
        --device cuda

Or specify a subset of routes directly::

    python scripts/run_all_torch_routes_candidate.py \
        --route-ids fixed_crop_ea_fbstcnet,sliding_window_srfnet \
        --data-root <path>

Candidate gate requirements (enforced by audit):
  - run_mode=candidate (no smoke/full_subjects)
  - no epoch override
  - test predictions (not val_only)
  - genuine score matrix (5 crop/window scores)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

DEFAULT_SEEDS = (42, 123, 456, 789, 1024)
DEFAULT_N_FOLDS = 5


def route_config_paths() -> list[Path]:
    return sorted((ROOT / "configs" / "routes" / "models").glob("*.yaml"))


def load_route(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"route config must be a mapping: {path}")
    return data


def route_config_by_id(route_id: str) -> Path | None:
    for p in route_config_paths():
        data = load_route(p)
        if data.get("route_id") == route_id:
            return p
    return None


def run_route_candidate(
    route_path: Path,
    *,
    data_root: Path,
    device: str = "auto",
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_folds: int = DEFAULT_N_FOLDS,
    output_dir: Path,
) -> dict[str, Any]:
    route_id = str(load_route(route_path).get("route_id") or route_path.stem)
    run_dir = output_dir / route_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_evaluation_protocol.py"),
        "--protocol", "p1",
        "--route-config", str(route_path),
        "--run-dir", str(run_dir),
        "--n-folds", str(n_folds),
        "--execute",
        "--execute-gate", "candidate",
        "--data-root", str(data_root),
        "--device", device,
    ]
    for s in seeds:
        cmd.extend(["--seed", str(s)])

    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    elapsed = time.perf_counter() - start

    result: dict[str, Any] = {
        "route_id": route_id,
        "route_config": route_path.relative_to(ROOT).as_posix(),
        "run_dir": str(run_dir),
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "stdout_tail": proc.stdout[-4000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-4000:] if proc.stderr else "",
    }

    try:
        lines = [l for l in proc.stdout.splitlines() if l.strip()]
        payload = json.loads(lines[-1]) if lines else {}
    except (json.JSONDecodeError, IndexError):
        payload = {}

    result["protocol_payload"] = payload
    result["passed"] = proc.returncode == 0 and payload.get("failed_jobs", 0) == 0
    return result


def _load_existing_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {}
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _load_existing_results(summary_path: Path) -> dict[str, dict[str, Any]]:
    summary = _load_existing_summary(summary_path)
    out: dict[str, dict[str, Any]] = {}
    for result in summary.get("results", []):
        if isinstance(result, dict) and result.get("route_id"):
            out[str(result["route_id"])] = result
    return out


def _resume_metadata_mismatches(summary: dict[str, Any], expected: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }


def _display_path(path: Path) -> str:
    resolved_root = ROOT.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch P1 candidate runner for torch_classifier routes.")
    parser.add_argument("--from-diagnostic", type=Path, help="Path to batch_summary.json from diagnostic run.")
    parser.add_argument("--route-ids", help="Comma-separated route IDs to run (alternative to --from-diagnostic).")
    parser.add_argument("--data-root", type=Path, required=True, help="HUST EEG .mat data root directory.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=lambda s: tuple(int(x) for x in s.split(",")), default=DEFAULT_SEEDS)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "batch_candidate")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip routes already marked passed in candidate_summary.json.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    route_ids: list[str] = []
    if args.from_diagnostic:
        summary = json.loads(args.from_diagnostic.read_text(encoding="utf-8"))
        for r in summary.get("results", []):
            if r.get("passed"):
                route_ids.append(str(r["route_id"]))
        print(f"Loaded {len(route_ids)} passed routes from {_display_path(args.from_diagnostic)}")
    elif args.route_ids:
        route_ids = [s.strip() for s in args.route_ids.split(",") if s.strip()]
    else:
        parser.error("--from-diagnostic or --route-ids is required")

    if not route_ids:
        print("No routes to run.")
        return 0

    route_paths: list[tuple[str, Path]] = []
    for rid in route_ids:
        path = route_config_by_id(rid)
        if path is None:
            print(f"WARNING: route config not found for {rid}, skipping")
            continue
        route_paths.append((rid, path))

    if args.dry_run:
        print(f"Would run {len(route_paths)} routes with {len(args.seeds)} seeds x {args.n_folds} folds:")
        for rid, _ in route_paths:
            print(f"  {rid}")
        return 0

    output_base = args.output_dir
    summary_path = output_base / "candidate_summary.json"
    output_base.mkdir(parents=True, exist_ok=True)
    resume_metadata = {
        "seeds": list(args.seeds),
        "n_folds": args.n_folds,
        "device": args.device,
        "data_root": str(args.data_root),
    }
    existing_summary = _load_existing_summary(summary_path) if not args.no_resume else {}
    resume_mismatches = _resume_metadata_mismatches(existing_summary, resume_metadata) if existing_summary else {}
    if resume_mismatches:
        keys = ", ".join(sorted(resume_mismatches))
        print(f"Existing candidate summary metadata mismatch ({keys}); rerunning matching routes.")
    existing_results = _load_existing_results(summary_path) if existing_summary and not resume_mismatches else {}

    results: list[dict[str, Any]] = []
    passed = 0
    failed = 0
    skipped_existing = 0

    def write_summary() -> None:
        summary: dict[str, Any] = {
            "total": len(route_paths),
            "passed": passed,
            "failed": failed,
            "skipped_existing": skipped_existing,
            "seeds": list(args.seeds),
            "n_folds": args.n_folds,
            "device": args.device,
            "data_root": str(args.data_root),
            "results": results,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for idx, (route_id, route_path) in enumerate(route_paths, 1):
        print(f"[{idx}/{len(route_paths)}] {route_id} ... ", end="", flush=True)
        existing = existing_results.get(route_id)
        if existing and existing.get("passed"):
            result = dict(existing)
            result["resume_status"] = "SKIPPED_EXISTING_PASS"
            results.append(result)
            passed += 1
            skipped_existing += 1
            print("SKIP existing PASS")
            write_summary()
            continue
        result = run_route_candidate(
            route_path,
            data_root=args.data_root,
            device=args.device,
            seeds=args.seeds,
            n_folds=args.n_folds,
            output_dir=output_base,
        )
        if resume_mismatches:
            result["resume_status"] = "RERUN_METADATA_MISMATCH"
            result["resume_mismatches"] = resume_mismatches
        results.append(result)
        if result["passed"]:
            passed += 1
            print(f"PASS ({result['elapsed_sec']}s)")
        else:
            failed += 1
            print(f"FAIL (rc={result['returncode']})")
        write_summary()
    print(f"\nCandidate summary: {passed} passed, {failed} failed  →  {_display_path(summary_path)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
