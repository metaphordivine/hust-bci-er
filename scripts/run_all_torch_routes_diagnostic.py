"""Batch-execute P1 diagnostic (smoke) runs for all torch_classifier routes.

Reads route configs from ``configs/routes/models/``, filters to
torch_classifier adapter routes, and runs each through
``run_evaluation_protocol.py`` with a single seed/fold smoke job.

Outputs a summary JSON to ``outputs/batch_diagnostic/`` recording
success/fail status, run_dir, and artifact presence for every route.

Usage::

    python scripts/run_all_torch_routes_diagnostic.py --data-root <path> [--device cuda]

The script skips score_fusion, toy_centroid, and toy-only routes
automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

DEFAULT_SEED = 42
DEFAULT_N_FOLDS = 2


def route_config_paths() -> list[Path]:
    return sorted((ROOT / "configs" / "routes" / "models").glob("*.yaml"))


def load_route(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"route config must be a mapping: {path}")
    return data


def route_adapter(route_data: dict[str, Any]) -> str:
    training = route_data.get("training")
    if isinstance(training, dict):
        return str(training.get("job_adapter") or "")
    return ""


def route_model_name(route_data: dict[str, Any]) -> str:
    model = route_data.get("model")
    if isinstance(model, dict):
        return str(model.get("name") or "")
    return ""


def is_torch_classifier_route(route_data: dict[str, Any]) -> bool:
    if route_adapter(route_data) != "torch_classifier":
        return False
    if route_model_name(route_data) in {"toy_centroid", "score_fusion"}:
        return False
    return True


def route_id_from_path(path: Path) -> str:
    data = load_route(path)
    return str(data.get("route_id") or path.stem)


def run_route_diagnostic(
    route_path: Path,
    *,
    data_root: Path | None = None,
    device: str = "auto",
    seed: int = DEFAULT_SEED,
    n_folds: int = DEFAULT_N_FOLDS,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    route_id = route_id_from_path(route_path)
    run_dir = output_dir or ROOT / "outputs" / "batch_diagnostic" / route_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_evaluation_protocol.py"),
        "--protocol", "p1",
        "--route-config", str(route_path),
        "--run-dir", str(run_dir),
        "--seed", str(seed),
        "--n-folds", str(n_folds),
        "--execute",
        "--execute-gate", "smoke",
        "--execute-epochs-override", "1",
        "--max-execute-jobs", "1",
        "--device", device,
    ]
    if data_root is not None:
        cmd.extend(["--data-root", str(data_root)])

    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    elapsed = time.perf_counter() - start

    result: dict[str, Any] = {
        "route_id": route_id,
        "route_config": route_path.relative_to(ROOT).as_posix(),
        "run_dir": str(run_dir),
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
    }

    try:
        payload = json.loads(proc.stdout.splitlines()[-1]) if proc.stdout.strip() else {}
    except (json.JSONDecodeError, IndexError):
        payload = {}

    result["protocol_payload"] = payload
    result["passed"] = proc.returncode == 0 and payload.get("failed_jobs", 0) == 0

    # Check for key artifacts
    job_runs = run_dir / "job_runs"
    if job_runs.exists():
        result["artifacts"] = _check_artifacts(job_runs)

    return result


def _check_artifacts(job_runs: Path) -> dict[str, bool]:
    artifacts = {
        "predictions_csv": False,
        "score_matrix_csv": False,
        "manifest_json": False,
        "audit_report_json": False,
    }
    for job_dir in sorted(job_runs.iterdir()):
        if not job_dir.is_dir():
            continue
        for name in artifacts:
            if (job_dir / f"{name.replace('_', '.')}").exists():
                artifacts[name] = True
                # Check the actual file name
        for fname in ("predictions.csv", "score_matrix.csv", "manifest.json", "audit_report.json"):
            if (job_dir / fname).exists():
                key = fname.replace(".", "_")
                artifacts[key] = True
    return artifacts


def _load_existing_results(summary_path: Path) -> dict[str, dict[str, Any]]:
    if not summary_path.exists():
        return {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for result in summary.get("results", []):
        if isinstance(result, dict) and result.get("route_id"):
            out[str(result["route_id"])] = result
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch P1 diagnostic runner for all torch_classifier routes.")
    parser.add_argument("--data-root", type=Path, help="HUST EEG .mat data root directory.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--output-dir", type=Path, help="Override default output directory.")
    parser.add_argument("--route-filter", help="Substring filter for route_id (optional).")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip routes already marked passed in batch_summary.json.")
    parser.add_argument("--dry-run", action="store_true", help="Print routes that would be executed and exit.")
    args = parser.parse_args(argv)

    routes = [
        p for p in route_config_paths() if is_torch_classifier_route(load_route(p))
    ]
    if args.route_filter:
        routes = [p for p in routes if args.route_filter in route_id_from_path(p)]

    if not routes:
        print("No torch_classifier routes found.")
        return 0

    print(f"Found {len(routes)} torch_classifier routes.")

    if args.dry_run:
        for p in routes:
            print(f"  {route_id_from_path(p)}  ({p.relative_to(ROOT)})")
        return 0

    data_root = args.data_root
    if data_root is None and os.environ.get("HUST_BCI_ER_DATA_ROOT"):
        data_root = Path(os.environ["HUST_BCI_ER_DATA_ROOT"])
    if data_root is None:
        parser.error("--data-root is required for executable torch diagnostics, or set HUST_BCI_ER_DATA_ROOT")

    output_base = args.output_dir or ROOT / "outputs" / "batch_diagnostic"
    summary_path = output_base / "batch_summary.json"
    output_base.mkdir(parents=True, exist_ok=True)
    existing_results = _load_existing_results(summary_path) if not args.no_resume else {}

    results: list[dict[str, Any]] = []
    passed = 0
    failed = 0
    skipped_existing = 0

    def write_summary() -> None:
        summary: dict[str, Any] = {
            "total": len(routes),
            "passed": passed,
            "failed": failed,
            "skipped_existing": skipped_existing,
            "seed": args.seed,
            "n_folds": args.n_folds,
            "device": args.device,
            "data_root": str(data_root),
            "results": results,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for idx, route_path in enumerate(routes, 1):
        route_id = route_id_from_path(route_path)
        print(f"[{idx}/{len(routes)}] {route_id} ... ", end="", flush=True)
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
        result = run_route_diagnostic(
            route_path,
            data_root=data_root,
            device=args.device,
            seed=args.seed,
            n_folds=args.n_folds,
            output_dir=output_base / route_id,
        )
        results.append(result)
        if result["passed"]:
            passed += 1
            print("PASS")
        else:
            failed += 1
            print(f"FAIL (rc={result['returncode']})")
        write_summary()
    print(f"\nSummary: {passed} passed, {failed} failed  ->  {summary_path.relative_to(ROOT)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
