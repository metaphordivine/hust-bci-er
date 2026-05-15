from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.config.docs import render_component_docs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate component registry documentation.")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "component_registry.md")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render_component_docs()
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != text:
            print(f"Component docs are not synchronized: {args.output}")
            return 1
        print("Component docs are synchronized.")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote component docs: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
