"""Ablation study runner for sliding-window route family.

Each ablation removes or varies one component relative to baseline.
Runs all variants as smoke (1 epoch), then compares primary metric.

Usage:
  python scripts/run_ablation.py \
    --route configs/routes/models/sliding_window_eegnet.yaml \
    --ablations configs/search/ablations/sliding_standard.yaml \
    --run-dir outputs/sliding_window_eegnet/ablation \
    --data-root $HUST_BCI_ER_DATA_ROOT
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _deep_set(target: dict, key_path: str, value: Any) -> None:
    keys = key_path.split(".")
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def _deep_del(target: dict, key_path: str) -> None:
    keys = key_path.split(".")
    for key in keys[:-1]:
        if not isinstance(target, dict):
            return
        if key not in target:
            return
        target = target[key]
    if isinstance(target, dict) and keys[-1] in target:
        del target[keys[-1]]


def _deep_get(target: dict, key_path: str) -> Any:
    keys = key_path.split(".")
    current: Any = target
    for key in keys:
        if not isinstance(current, dict):
            raise KeyError(key_path)
        current = current[key]
    return current


def _build_temp_route_config(
    base_route: dict,
    work_dir: Path,
    route_id: str,
    *,
    overrides: dict[str, Any] | None = None,
    removes: list[str] | None = None,
) -> Path:
    """Build a temporary route config whose route_id matches its file stem."""
    work_dir.mkdir(parents=True, exist_ok=True)
    route = copy.deepcopy(base_route)
    route["route_id"] = route_id
    for key_path, value in (overrides or {}).items():
        _deep_set(route, key_path, value)
    for key_path in removes or []:
        _deep_del(route, key_path)
    path = work_dir / f"{route_id}.yaml"
    path.write_text(yaml.safe_dump(route, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _build_ablated_config(
    base_route: dict,
    ablation: dict[str, Any],
    work_dir: Path,
    variant_name: str,
) -> Path:
    """Apply ablation overrides to build a temp route config."""
    return _build_temp_route_config(
        base_route,
        work_dir,
        f"ablation_{variant_name}",
        overrides=ablation.get("overrides", {}),
        removes=ablation.get("remove", []),
    )


def _run_single(
    *,
    temp_route: Path,
    run_dir: Path,
    data_root: str,
    seed: int = 42,
    device: str = "auto",
    smoke_epochs: int = 1,
) -> dict[str, Any]:
    """Run a single training job and return parsed metrics."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["HUST_BCI_ER_SEED"] = str(seed)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["HUST_BCI_ER_DATA_ROOT"] = data_root

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_route.py"),
        "--mode", "smoke",
        "--route", str(temp_route),
        "--run-dir", str(run_dir),
        "--smoke-epochs", str(smoke_epochs),
        "--seed", str(seed),
        "--device", device,
    ]

    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        return {
            "error": True,
            "returncode": result.returncode,
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }

    stdout = result.stdout.strip()
    lines = stdout.splitlines()
    json_line = None
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{"):
            json_line = line
            break
    if json_line is None:
        return {"error": True, "message": "no JSON output found", "stdout": stdout[-1000:]}

    try:
        parsed = json.loads(json_line)
    except json.JSONDecodeError:
        return {"error": True, "message": "invalid JSON output", "stdout": json_line[:1000]}

    return dict(parsed)


def _has_value(raw: Any) -> bool:
    return raw is not None and raw != ""


def _coerce_metric_value(raw: Any) -> float | None:
    if not _has_value(raw):
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            raw = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                return float(text)
            except ValueError:
                return None
    if isinstance(raw, dict):
        for value in raw.values():
            parsed = _coerce_metric_value(value)
            if parsed is not None:
                return parsed
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _extract_metric_value(primary_metric: Any) -> float:
    """Extract a single float from the primary_metric dict/value."""
    value = _coerce_metric_value(primary_metric)
    if value is None:
        raise ValueError("missing or invalid numeric primary_metric")
    return value


def _mark_metric_error_if_needed(row: dict[str, Any]) -> dict[str, Any]:
    """Treat malformed metrics as failed runs instead of valid 0.0 scores."""
    if _is_error_result(row):
        return row
    try:
        _extract_metric_value(row.get("primary_metric"))
    except ValueError as exc:
        row["error"] = True
        row["message"] = str(exc)
    return row


def _is_error_result(row: dict[str, Any]) -> bool:
    raw = row.get("error", False)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return False
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y"}
    return bool(raw)


def _write_results_csv(results: list[dict[str, Any]], csv_path: Path, baseline_metric: float | None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["variant", "description", "error", "metric_value", "delta_vs_baseline", "error_detail"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            r = _mark_metric_error_if_needed(r)
            is_error = _is_error_result(r)
            variant_metric = None
            error_detail = ""
            if is_error:
                error_detail = r.get("stderr", r.get("message", ""))[:500]
            else:
                variant_metric = _extract_metric_value(r.get("primary_metric"))
            writer.writerow({
                "variant": r.get("variant", ""),
                "description": r.get("description", ""),
                "error": str(is_error),
                "metric_value": "" if variant_metric is None else f"{variant_metric:.6f}",
                "delta_vs_baseline": "" if variant_metric is None or baseline_metric is None else f"{variant_metric - baseline_metric:.6f}",
                "error_detail": error_detail,
            })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run ablation study variants against a baseline route."
    )
    parser.add_argument("--route", type=Path, required=True, help="Base route config YAML.")
    parser.add_argument("--ablations", type=Path, required=True, help="YAML defining ablation variants.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--data-root", required=True, help="Path to .mat data root.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Print variants without running.")
    args = parser.parse_args(argv)

    base_route = yaml.safe_load(args.route.read_text(encoding="utf-8")) or {}
    if not isinstance(base_route, dict):
        raise ValueError(f"invalid route config: {args.route}")

    ablation_spec = yaml.safe_load(args.ablations.read_text(encoding="utf-8")) or {}
    if not isinstance(ablation_spec, dict):
        raise ValueError(f"invalid ablation spec: {args.ablations}")

    variants = ablation_spec.get("variants", [])
    if not variants:
        raise ValueError("ablation spec must have 'variants' list")

    run_dir = args.run_dir.resolve()
    work_dir = run_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"Baseline: {args.route}")
        for v in variants:
            name = v.get("name", "unnamed")
            desc = v.get("description", "")
            overrides = v.get("overrides", {})
            removes = v.get("remove", [])
            print(f"  {name}: {desc}")
            if overrides:
                print(f"    overrides: {overrides}")
            if removes:
                print(f"    remove: {removes}")
        return 0

    results: list[dict[str, Any]] = []

    # Baseline run
    print("[baseline] Running baseline ... ", end="", flush=True)
    baseline_dir = work_dir / "baseline"
    baseline_route_path = _build_temp_route_config(base_route, work_dir, "baseline_route")
    baseline_result = _run_single(
        temp_route=baseline_route_path,
        run_dir=baseline_dir,
        data_root=args.data_root,
        seed=args.seed,
        device=args.device,
        smoke_epochs=args.smoke_epochs,
    )
    baseline_result["variant"] = "baseline"
    baseline_result["description"] = "Baseline (no changes)"
    baseline_result = _mark_metric_error_if_needed(baseline_result)
    results.append(baseline_result)
    if _is_error_result(baseline_result):
        print(f"FAILED: {baseline_result.get('stderr', baseline_result.get('message', ''))[:120]}")
        csv_path = run_dir / "ablation_results.csv"
        _write_results_csv(results, csv_path, baseline_metric=None)
        print("Baseline failed; stopping ablation because deltas would be invalid.")
        print(f"Results saved to: {csv_path}")
        return 1
    else:
        pm = baseline_result.get("primary_metric", {})
        print(f"OK  {pm}")

    baseline_metric = _extract_metric_value(baseline_result.get("primary_metric"))

    # Run each ablation variant
    for variant in variants:
        name = variant.get("name", "unnamed")
        desc = variant.get("description", "")
        print(f"[{name}] {desc} ... ", end="", flush=True)

        variant_dir = work_dir / f"ablation_{name}"
        try:
            temp_route = _build_ablated_config(base_route, variant, work_dir, name)
            result = _run_single(
                temp_route=temp_route,
                run_dir=variant_dir,
                data_root=args.data_root,
                seed=args.seed,
                device=args.device,
                smoke_epochs=args.smoke_epochs,
            )
        except Exception as exc:
            result = {"error": True, "stderr": str(exc)}

        result["variant"] = name
        result["description"] = desc
        result = _mark_metric_error_if_needed(result)
        results.append(result)

        if _is_error_result(result):
            print(f"FAILED: {result.get('stderr', result.get('message', ''))[:120]}")
        else:
            pm = result.get("primary_metric", {})
            variant_metric = _extract_metric_value(result.get("primary_metric"))
            delta = variant_metric - baseline_metric
            print(f"OK  {pm}  (delta={delta:+.4f})")

    # Write comparison table
    csv_path = run_dir / "ablation_results.csv"
    _write_results_csv(results, csv_path, baseline_metric)

    successful = [r for r in results if not _is_error_result(r)]
    failed = len(results) - len(successful)
    print(f"\n--- Ablation complete ---")
    print(f"Total: {len(results)}, Success: {len(successful)}, Failed: {failed}")
    print(f"Baseline metric: {baseline_metric:.6f}")
    if successful:
        for r in successful:
            if r.get("variant") == "baseline":
                continue
            vm = _extract_metric_value(r.get("primary_metric"))
            print(f"  {r['variant']}: {vm:.6f}  (delta={vm - baseline_metric:+.6f})")
    print(f"Results saved to: {csv_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
