from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTE_NAME = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*\.yaml$")
SUMMARY_NAME = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*_summary\.md$")
BAD_SCRIPT = re.compile(r"(final_final|try_again|debug_|new_test|run\d+)", re.IGNORECASE)


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


def main() -> int:
    errors: list[str] = []

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

    for path in (ROOT / "scripts").glob("*.py"):
        if BAD_SCRIPT.search(path.name):
            errors.append(f"unstable script name is not allowed in scripts/: {path.name}")

    files = tracked_files()
    if files is not None:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
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
