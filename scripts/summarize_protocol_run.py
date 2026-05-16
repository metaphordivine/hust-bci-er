from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.evaluation.protocols.summary import write_protocol_summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize completed protocol runner job outputs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    audit = write_protocol_summary(args.run_dir)
    print(json.dumps(audit, ensure_ascii=False))
    return 0 if audit["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
