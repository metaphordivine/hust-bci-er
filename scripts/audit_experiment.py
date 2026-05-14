from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from hust_bci_er.audit.manifest import validate_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args(argv)

    failures: list[str] = []

    validate_route = ROOT / "scripts" / "validate_route.py"
    route_status = __import__("subprocess").run(
        [sys.executable, str(validate_route), str(args.route)],
        cwd=ROOT,
    )
    if route_status.returncode != 0:
        failures.append("route config validation failed")

    if args.run is not None:
        manifest_path = args.run / "manifest.json"
        failures.extend(validate_manifest(manifest_path))
    else:
        print("No run directory provided; manifest and metric recompute checks skipped.")

    if failures:
        print("Experiment audit failed:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("Experiment audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

