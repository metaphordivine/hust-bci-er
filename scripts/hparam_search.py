"""Coarse-to-fine hyperparameter grid search for route configs.

Usage:
  # Coarse stage
  python scripts/hparam_search.py \\
    --route configs/routes/models/sliding_window_eegnet.yaml \\
    --stage coarse \\
    --search-space configs/search/spaces/sliding_baseline.yaml \\
    --run-dir outputs/sliding_window_eegnet/hparam_search \\
    --data-root $HUST_BCI_ER_DATA_ROOT

  # Fine stage (reads coarse results, narrows grid around best)
  python scripts/hparam_search.py \\
    --route configs/routes/models/sliding_window_eegnet.yaml \\
    --stage fine \\
    --search-space configs/search/spaces/sliding_baseline.yaml \\
    --run-dir outputs/sliding_window_eegnet/hparam_search \\
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
    """Set a nested dict value by dot-separated key path."""
    keys = key_path.split(".")
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def _deep_get(target: dict, key_path: str) -> Any:
    """Get a nested dict value by dot-separated key path."""
    keys = key_path.split(".")
    current: Any = target
    for key in keys:
        if not isinstance(current, dict):
            raise KeyError(key_path)
        current = current[key]
    return current


def _generate_coarse_combinations(search_space: dict) -> list[dict[str, Any]]:
    """Cartesian product of coarse grid values."""
    params = search_space.get("parameters", {})
    if not params:
        raise ValueError("search_space must have a 'parameters' mapping")

    keys: list[str] = []
    value_lists: list[list] = []
    for key, spec in params.items():
        values = spec.get("coarse") if isinstance(spec, dict) else spec
        if not values:
            continue
        keys.append(key)
        value_lists.append(list(values))

    if not keys:
        raise ValueError("no parameters with coarse values in search space")

    combinations: list[dict[str, Any]] = []
    _cartesian_product(keys, value_lists, 0, {}, combinations)
    return combinations


def _generate_fine_combinations(
    search_space: dict,
    best_combo: dict[str, Any],
    *,
    factor: float = 0.5,
) -> list[dict[str, Any]]:
    """Generate fine grid around best combination.

    For each parameter, creates a narrower range centered on the best value.
    If the parameter is numeric and search_space defines a fine grid explicitly,
    uses that. Otherwise, generates [best*(1-factor), best, best*(1+factor)]
    for continuous params or adjacent discrete values.
    """
    params = search_space.get("parameters", {})
    keys: list[str] = []
    value_lists: list[list] = []
    for key, spec in params.items():
        fine_values = spec.get("fine") if isinstance(spec, dict) else None
        if fine_values:
            keys.append(key)
            value_lists.append(list(fine_values))
            continue

        # Auto-generate fine range around best value
        best_val = best_combo.get(key)
        if best_val is None:
            continue
        coarse_values = spec.get("coarse") if isinstance(spec, dict) else spec
        if not coarse_values:
            continue

        keys.append(key)
        if isinstance(best_val, (int, float)):
            half_step = abs(best_val) * factor
            if half_step == 0:
                half_step = best_val * 0.25 if best_val != 0 else 0.1
            candidates = sorted(set([
                best_val - half_step,
                best_val,
                best_val + half_step,
            ]))
            # Filter to sensible ranges (no negative lr etc.)
            candidates = [c for c in candidates if c > 0]
            if not candidates:
                candidates = [best_val]
            value_lists.append(candidates)
        else:
            # Discrete/non-numeric: just use best
            value_lists.append([best_val])

    combinations: list[dict[str, Any]] = []
    _cartesian_product(keys, value_lists, 0, {}, combinations)
    return combinations


def _cartesian_product(
    keys: list[str],
    value_lists: list[list],
    depth: int,
    current: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    if depth == len(keys):
        results.append(dict(current))
        return
    for val in value_lists[depth]:
        current[keys[depth]] = val
        _cartesian_product(keys, value_lists, depth + 1, current, results)


def _combo_id(index: int, combo: dict[str, Any]) -> str:
    """Short identifier for a parameter combination."""
    parts = [f"{k.split('.')[-1]}={v}" for k, v in sorted(combo.items())]
    return f"run_{index:03d}__" + "__".join(parts)


def _build_temp_route(base_route: dict, overrides: dict[str, Any], work_dir: Path, route_id: str = "temp_route") -> Path:
    """Write a temporary route YAML with overridden parameter values."""
    work_dir.mkdir(parents=True, exist_ok=True)
    route = copy.deepcopy(base_route)
    route["route_id"] = route_id
    for key_path, value in overrides.items():
        _deep_set(route, key_path, value)
    path = work_dir / f"{route_id}.yaml"
    path.write_text(yaml.safe_dump(route, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


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

    # Parse JSON from last line of stdout
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


def _read_coarse_results(results_csv: Path) -> list[dict[str, Any]]:
    """Read previous coarse results for fine stage refinement."""
    if not results_csv.exists():
        raise FileNotFoundError(f"coarse results not found: {results_csv}")
    rows: list[dict[str, Any]] = []
    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _is_error_result(row: dict[str, Any]) -> bool:
    raw = row.get("error", False)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return False
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y"}
    return bool(raw)


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


def _metric_value(row: dict[str, Any], metric_key: str = "primary_metric") -> float:
    for key in (metric_key, "metric_value"):
        if key in row:
            value = _coerce_metric_value(row.get(key))
            if value is not None:
                return value
    metric_columns = sorted(
        key for key, value in row.items()
        if key.startswith("metric_") and key != "metric_value" and _has_value(value)
    )
    for key in metric_columns:
        value = _coerce_metric_value(row.get(key))
        if value is not None:
            return value
    raise ValueError("successful result row has no numeric metric value")


def _select_best(results: list[dict[str, Any]], metric_key: str = "primary_metric") -> dict[str, Any]:
    """Select best result by primary metric value (higher is better)."""
    valid = [r for r in results if not _is_error_result(r)]
    if not valid:
        raise ValueError("no successful results to select best from")
    return max(valid, key=lambda row: _metric_value(row, metric_key))


def _results_to_csv(results: list[dict[str, Any]], path: Path, param_keys: list[str]) -> None:
    """Write results to CSV, extracting nested metrics into flat columns."""
    if not results:
        return
    # Build flat fieldnames
    flat_results: list[dict[str, Any]] = []
    for r in results:
        flat: dict[str, Any] = {}
        for k in param_keys:
            flat[k] = r.get("combo", {}).get(k, "")
        is_error = _is_error_result(r)
        flat["error"] = is_error
        if is_error:
            flat["stderr"] = r.get("stderr", "")[:500]
        else:
            flat["run_dir"] = r.get("run_dir", "")
        # Flatten primary_metric dict
        pm = r.get("primary_metric", {})
        if isinstance(pm, dict):
            metric_values = list(pm.values())
            if metric_values:
                flat["metric_value"] = metric_values[0]
            for mk, mv in pm.items():
                flat[f"metric_{mk}"] = mv
        elif pm not in {None, ""}:
            flat["metric_value"] = pm
        flat_results.append(flat)

    preferred = [*param_keys, "error", "run_dir", "metric_value"]
    metric_keys = sorted({key for row in flat_results for key in row if key.startswith("metric_") and key != "metric_value"})
    trailing = ["stderr"]
    all_keys: list[str] = []
    for key in [*preferred, *metric_keys, *trailing]:
        if key not in all_keys and any(key in row for row in flat_results):
            all_keys.append(key)
    for row in flat_results:
        for key in row:
            if key not in all_keys:
                all_keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(flat_results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Coarse-to-fine hyperparameter grid search for route training."
    )
    parser.add_argument("--route", type=Path, required=True, help="Base route config YAML.")
    parser.add_argument("--stage", choices=["coarse", "fine"], required=True)
    parser.add_argument("--search-space", type=Path, required=True, help="YAML defining parameter grids.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--data-root", required=True, help="Path to .mat data root.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Print combinations without running.")
    args = parser.parse_args(argv)

    base_route = yaml.safe_load(args.route.read_text(encoding="utf-8")) or {}
    if not isinstance(base_route, dict):
        raise ValueError(f"invalid route config: {args.route}")

    search_space = yaml.safe_load(args.search_space.read_text(encoding="utf-8")) or {}
    if not isinstance(search_space, dict):
        raise ValueError(f"invalid search space: {args.search_space}")

    run_dir = args.run_dir.resolve()
    work_dir = run_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "coarse":
        combinations = _generate_coarse_combinations(search_space)
        print(f"Coarse grid: {len(combinations)} combinations")
    else:
        coarse_csv = run_dir / "coarse_results.csv"
        coarse_results = _read_coarse_results(coarse_csv)
        # Reconstruct best combo from CSV
        best = _select_best(coarse_results)
        param_keys = [k for k in search_space.get("parameters", {}) if k in best]
        best_combo = {k: best.get(k) for k in param_keys}
        # Convert string values back to original types
        for k, v in best_combo.items():
            try:
                best_combo[k] = float(v) if "." in str(v) or "e" in str(v).lower() else int(v)
            except (ValueError, TypeError):
                best_combo[k] = v
        combinations = _generate_fine_combinations(search_space, best_combo)
        print(f"Fine grid (around best={best_combo}): {len(combinations)} combinations")

    if args.dry_run:
        for i, combo in enumerate(combinations):
            print(f"  {_combo_id(i, combo)}: {combo}")
        return 0

    param_keys = list(combinations[0].keys()) if combinations else []

    results: list[dict[str, Any]] = []
    for i, combo in enumerate(combinations):
        combo_id = _combo_id(i, combo)
        combo_run_dir = work_dir / combo_id
        print(f"[{i+1}/{len(combinations)}] {combo_id} ... ", end="", flush=True)

        safe_id = combo_id.replace(".", "p").replace("=", "_").replace("[", "").replace("]", "").replace(",", "_").replace(" ", "")
        try:
            temp_route = _build_temp_route(base_route, combo, combo_run_dir, route_id=safe_id)
            result = _run_single(
                temp_route=temp_route,
                run_dir=combo_run_dir,
                data_root=args.data_root,
                seed=args.seed,
                device=args.device,
                smoke_epochs=args.smoke_epochs,
            )
        except Exception as exc:
            result = {"error": True, "stderr": str(exc)}

        result["combo"] = combo
        results.append(result)

        if _is_error_result(result):
            err_msg = result.get("stderr", result.get("message", "unknown"))[:120]
            print(f"FAILED: {err_msg}")
        else:
            pm = result.get("primary_metric", {})
            if isinstance(pm, dict):
                metric_str = ", ".join(f"{k}={v:.4f}" for k, v in pm.items())
            else:
                metric_str = str(pm)
            print(f"OK  {metric_str}")

    # Save results
    csv_path = run_dir / f"{args.stage}_results.csv"
    _results_to_csv(results, csv_path, param_keys)

    # Print summary
    successful = [r for r in results if not _is_error_result(r)]
    failed = len(results) - len(successful)
    print(f"\n--- {args.stage} stage complete ---")
    print(f"Total: {len(results)}, Success: {len(successful)}, Failed: {failed}")
    if successful:
        best = _select_best(successful)
        print(f"Best: {best.get('combo')}")
        print(f"Best metric value: {_metric_value(best):.6f}")
    print(f"Results saved to: {csv_path}")

    # Save best combo as JSON for fine stage
    if successful:
        best_path = run_dir / f"{args.stage}_best.json"
        best_path.write_text(json.dumps(best, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return 0 if successful and failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
