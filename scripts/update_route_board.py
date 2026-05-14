from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BOARD_PATH = ROOT / "reports" / "route_board.md"


def route_files() -> list[Path]:
    return sorted((ROOT / "configs" / "routes" / "models").glob("*.yaml"))


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"route config must be a mapping: {path}")
    return data


def model_name(model: Any) -> str:
    if isinstance(model, str):
        return model
    if isinstance(model, dict):
        return str(model.get("name") or "")
    return ""


def protocol_name(evaluation: Any) -> str:
    if isinstance(evaluation, str):
        return evaluation
    if isinstance(evaluation, dict):
        return str(evaluation.get("protocol") or "")
    return ""


def generate_board() -> str:
    lines = [
        "# Route Board",
        "",
        "generated_by: `python scripts/update_route_board.py`",
        "",
        "| route_id | status | model | protocol | summary |",
        "|---|---|---|---|---|",
    ]
    for path in route_files():
        data = load_yaml(path)
        route_id = str(data.get("route_id") or path.stem)
        status = str(data.get("status") or "")
        model = model_name(data.get("model"))
        protocol = protocol_name(data.get("evaluation"))
        summary = ROOT / "reports" / "route_summaries" / f"{route_id}_summary.md"
        summary_state = "present" if summary.exists() else "missing"
        lines.append(f"| `{route_id}` | {status} | `{model}` | `{protocol}` | {summary_state} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    expected = generate_board()
    if args.check:
        actual = BOARD_PATH.read_text(encoding="utf-8") if BOARD_PATH.exists() else ""
        if actual != expected:
            print("Route board is not synchronized. Run: python scripts/update_route_board.py", file=sys.stderr)
            return 1
        print("Route board is synchronized.")
        return 0

    BOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOARD_PATH.write_text(expected, encoding="utf-8")
    print(f"Updated {BOARD_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
