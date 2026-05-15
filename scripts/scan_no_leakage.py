from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.audit.source_scanner import scan_no_leakage  # noqa: E402


def main() -> int:
    findings = scan_no_leakage(ROOT)
    if findings:
        print("No-leakage source scanner failed:")
        for item in findings:
            print(f"  - {item.path}:{item.line}: {item.pattern}")
        return 1
    print("No-leakage source scanner passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
