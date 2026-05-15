from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ROUTE_NAME = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*\.yaml$")
SUMMARY_NAME = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*_summary\.md$")
BAD_SCRIPT = re.compile(r"(final_final|try_again|debug_|new_test|run\d+)", re.IGNORECASE)
SLIDING_WINDOW_ROUTES = {
    "sliding_window_conformer_lite": "conformer_lite",
    "sliding_window_deformer_lite": "deformer_lite",
    "sliding_window_eegnet": "eegnet",
    "sliding_window_srfnet": "srfnet",
}


def tracked_files() -> list[Path] | None:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    return [ROOT / line for line in result.stdout.splitlines() if line.strip()]


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def without_variant_fields(data: dict) -> dict:
    clone = dict(data)
    clone.pop("route_id", None)
    model = clone.get("model")
    if isinstance(model, dict):
        model = dict(model)
        model.pop("name", None)
        clone["model"] = model
    return clone


def sliding_window_family_errors() -> list[str]:
    errors: list[str] = []
    routes = {}
    for route_id, model_name in SLIDING_WINDOW_ROUTES.items():
        path = ROOT / "configs" / "routes" / "models" / f"{route_id}.yaml"
        if not path.exists():
            errors.append(f"sliding-window family route is missing: {path.relative_to(ROOT)}")
            continue
        data = load_yaml(path)
        routes[route_id] = data
        model = data.get("model")
        actual_model = model.get("name") if isinstance(model, dict) else None
        if actual_model != model_name:
            errors.append(f"{route_id} model.name must be {model_name}, got {actual_model}")
    if len(routes) == len(SLIDING_WINDOW_ROUTES):
        reference_id = "sliding_window_eegnet"
        reference = without_variant_fields(routes[reference_id])
        for route_id, data in sorted(routes.items()):
            if route_id == reference_id:
                continue
            if without_variant_fields(data) != reference:
                errors.append(f"{route_id} differs from sliding-window family baseline outside route_id/model.name")
    return errors


def main() -> int:
    errors: list[str] = []

    required_files = [
        "README.md",
        "AGENTS.md",
        "docs/目录怎么用.md",
        "agent_protocols/experiment_audit.md",
        "configs/statuses.yaml",
        "configs/audit_decisions.yaml",
    ]
    for rel in required_files:
        if not (ROOT / rel).exists():
            errors.append(f"required governance file is missing: {rel}")
    if not (ROOT / "configs" / "splits").is_dir():
        errors.append("required split manifest directory is missing: configs/splits")
    if not (ROOT / "configs" / "datasets").is_dir():
        errors.append("required dataset manifest directory is missing: configs/datasets")

    routes_root = ROOT / "configs" / "routes"
    tier_route_dirs = {"stable", "candidates", "deprecated", "experimental"}
    for name in tier_route_dirs:
        if (routes_root / name).exists():
            errors.append(f"route tier directory is not allowed: configs/routes/{name}")

    for path in routes_root.glob("**/*.yaml"):
        if not ROUTE_NAME.match(path.name):
            errors.append(f"route config file must be snake_case: {path.relative_to(ROOT)}")

    for path in (ROOT / "reports" / "route_summaries").glob("*.md"):
        if not SUMMARY_NAME.match(path.name):
            errors.append(f"route summary file must be <route_id>_summary.md: {path.relative_to(ROOT)}")
        route_id = path.name.removesuffix("_summary.md")
        if f"route_id: {route_id}" not in path.read_text(encoding="utf-8", errors="ignore"):
            errors.append(f"route summary must include matching route_id: {path.relative_to(ROOT)}")

    for path in (ROOT / "scripts").glob("*.py"):
        if BAD_SCRIPT.search(path.name):
            errors.append(f"unstable script name is not allowed in scripts/: {path.name}")

    errors.extend(sliding_window_family_errors())

    files = tracked_files()
    if files is not None:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            if " " in rel:
                errors.append(f"tracked path must not contain spaces: {rel}")
            if rel.startswith(("outputs/", "scratch/")):
                errors.append(f"local artifact path is tracked: {rel}")
            if any(part in rel.lower() for part in ["checkpoint", ".ckpt", ".pth", ".pt"]):
                errors.append(f"checkpoint-like artifact is tracked: {rel}")

    if errors:
        print("Repository convention check failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("Repository convention check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
