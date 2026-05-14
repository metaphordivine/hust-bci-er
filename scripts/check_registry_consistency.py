from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.config.consistency import registry_consistency_errors  # noqa: E402


def main() -> int:
    errors = registry_consistency_errors(ROOT)
    if errors:
        print("Registry consistency check failed:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("Registry consistency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
