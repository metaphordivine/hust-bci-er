"""Orchestrate score_fusion route assembly from component route predictions.

For each score_fusion route in ``configs/routes/models/``, this script:

1. Resolves each logical component ID to a base torch_classifier route via
   the component-route map (``hust_bci_er.config.component_map``).
2. Looks for existing component-score CSV files under a well-known
   ``component_scores/`` directory, or exports them from the base route's
   predictions if ``--export-missing`` is passed.
3. Calls ``assemble_score_route.py`` to produce the fused predictions CSV.
4. Optionally runs ``repo_doctor.py experiment`` audit on the result.

Usage::

    # List score_fusion routes and their component bindings (dry-run):
    python scripts/run_score_fusion_routes.py --dry-run

    # Assemble from pre-exported component scores:
    python scripts/run_score_fusion_routes.py --component-scores-dir outputs/component_scores

    # Auto-export component scores from base route run dirs, then assemble:
    python scripts/run_score_fusion_routes.py \
        --component-scores-dir outputs/component_scores \
        --base-runs-dir outputs/candidate_all_models \
        --export-missing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from hust_bci_er.config.component_map import base_route_for_component  # noqa: E402
from hust_bci_er.inference.clean_score_routes import score_route_by_id  # noqa: E402


def route_config_paths() -> list[Path]:
    return sorted((ROOT / "configs" / "routes" / "models").glob("*.yaml"))


def load_route(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"route config must be a mapping: {path}")
    return data


def route_model_name(route_data: dict[str, Any]) -> str:
    model = route_data.get("model")
    if isinstance(model, dict):
        return str(model.get("name") or "")
    return ""


def is_score_fusion_route(route_data: dict[str, Any]) -> bool:
    return route_model_name(route_data) == "score_fusion"


def score_fusion_routes() -> list[tuple[Path, dict[str, Any]]]:
    result: list[tuple[Path, dict[str, Any]]] = []
    for path in route_config_paths():
        data = load_route(path)
        if is_score_fusion_route(data):
            result.append((path, data))
    return result


def find_predictions_in_run_dir(run_dir: Path) -> Path | None:
    candidate = run_dir / "predictions.csv"
    if candidate.exists():
        return candidate
    for item in sorted(run_dir.glob("**/predictions.csv")):
        return item
    return None


def export_component_score(
    component_id: str,
    base_route_id: str,
    predictions_path: Path,
    output_path: Path,
) -> int:
    route_config = ROOT / "configs" / "routes" / "models" / f"{base_route_id}.yaml"
    if not route_config.exists():
        print(f"  base route config not found for {component_id}: {route_config}", file=sys.stderr)
        return 1
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_component_scores.py"),
        "--route", str(route_config),
        "--predictions", str(predictions_path),
        "--component-id", component_id,
        "--output", str(output_path),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        print(f"  export failed for {component_id}: {proc.stderr[-500:]}", file=sys.stderr)
    return proc.returncode


def assemble_score_fusion(
    route_path: Path,
    component_score_paths: dict[str, Path],
    output_dir: Path,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "assemble_score_route.py"),
        "--route", str(route_path),
        "--output", str(predictions_path),
    ]
    for component_id, path in component_score_paths.items():
        cmd.extend(["--component-score", f"{component_id}={path}"])

    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        print(f"  assembly failed: {proc.stderr[-500:]}", file=sys.stderr)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Orchestrate score_fusion route assembly.")
    parser.add_argument("--dry-run", action="store_true", help="Print score_fusion routes and exit.")
    parser.add_argument("--component-scores-dir", type=Path, help="Directory of pre-exported component score CSVs.")
    parser.add_argument("--base-runs-dir", type=Path, help="Directory containing per-route run dirs with predictions.csv.")
    parser.add_argument("--export-missing", action="store_true", help="Auto-export component scores from base route predictions.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "score_fusion_batch")
    parser.add_argument("--route-filter", help="Substring filter for score_fusion route_id.")
    parser.add_argument("--audit", action="store_true", help="Run repo_doctor experiment audit after assembly.")
    args = parser.parse_args(argv)

    routes = score_fusion_routes()
    if args.route_filter:
        routes = [(p, d) for p, d in routes if args.route_filter in str(d.get("route_id", ""))]

    if not routes:
        print("No score_fusion routes found.")
        return 0

    if args.dry_run:
        print(f"Found {len(routes)} score_fusion routes:\n")
        for route_path, data in routes:
            route_id = data["route_id"]
            score_route = score_route_by_id(str(route_id))
            components = list(data.get("model", {}).get("components", []))
            print(f"  {route_id}")
            for comp in components:
                base = base_route_for_component(str(comp))
                status = f"→ {base}" if base else "→ (no torch route)"
                print(f"    {comp} {status}")
            if score_route and score_route.weights:
                print(f"    weights: {score_route.weights}")
            print()
        return 0

    if not args.component_scores_dir and not (args.base_runs_dir and args.export_missing):
        parser.error("--component-scores-dir is required (or use --base-runs-dir --export-missing)")

    component_scores_dir = args.component_scores_dir or (args.output_dir / "component_scores")
    component_scores_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    passed = 0
    failed = 0

    for idx, (route_path, data) in enumerate(routes, 1):
        route_id = str(data["route_id"])
        components = [str(c) for c in data.get("model", {}).get("components", [])]
        print(f"[{idx}/{len(routes)}] {route_id}")

        component_scores: dict[str, Path] = {}
        missing: list[str] = []

        for comp_id in components:
            expected = component_scores_dir / f"{comp_id}.csv"
            if expected.exists():
                component_scores[comp_id] = expected
                continue

            # Try to export from base route predictions
            base_route = base_route_for_component(comp_id)
            if base_route is None:
                print(f"  {comp_id}: no base route mapping (skip)")
                missing.append(comp_id)
                continue

            if args.export_missing and args.base_runs_dir:
                base_run_dir = args.base_runs_dir / base_route
                predictions = find_predictions_in_run_dir(base_run_dir)
                if predictions is None:
                    print(f"  {comp_id}: no predictions found in {base_run_dir}")
                    missing.append(comp_id)
                    continue
                rc = export_component_score(comp_id, base_route, predictions, expected)
                if rc == 0:
                    component_scores[comp_id] = expected
                    print(f"  {comp_id}: exported from {predictions.relative_to(ROOT)}")
                else:
                    missing.append(comp_id)
            else:
                print(f"  {comp_id}: component score not found at {expected}")
                missing.append(comp_id)

        if missing:
            print(f"  SKIP: missing component scores: {missing}")
            results.append({"route_id": route_id, "status": "SKIP", "missing_components": missing})
            failed += 1
            continue

        output_dir = args.output_dir / route_id
        rc = assemble_score_fusion(route_path, component_scores, output_dir)
        result: dict[str, Any] = {
            "route_id": route_id,
            "run_dir": str(output_dir),
            "status": "PASS" if rc == 0 else "FAIL",
        }

        if rc == 0 and args.audit:
            audit_proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "repo_doctor.py"),
                    "experiment",
                    "--route", str(route_path),
                    "--run", str(output_dir),
                    "--gate", "candidate",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            result["audit_returncode"] = audit_proc.returncode
            result["status"] = "PASS" if audit_proc.returncode == 0 else "AUDIT_FAIL"

        if result["status"] == "PASS":
            passed += 1
            print(f"  PASS  →  {output_dir.relative_to(ROOT)}")
        else:
            failed += 1
            print(f"  {result['status']}")

        results.append(result)

    summary_path = args.output_dir / "score_fusion_summary.json"
    summary = {
        "total": len(routes),
        "passed": passed,
        "failed": failed,
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nScore fusion summary: {passed} passed, {failed} failed  →  {summary_path.relative_to(ROOT)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
