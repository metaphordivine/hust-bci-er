from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.audit.promotion import check_candidate_audit, check_promotion_fields, parse_key_value_markdown  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check promotion audit fields and candidate report binding.")
    parser.add_argument("--promotion-audit", type=Path, required=True)
    args = parser.parse_args(argv)
    fields = parse_key_value_markdown(args.promotion_audit)
    errors = check_promotion_fields(fields)
    if fields.get("candidate_audit_report") and fields.get("route_id"):
        errors.extend(check_candidate_audit(ROOT / fields["candidate_audit_report"], route_id=fields["route_id"]))
    if errors:
        print("Promotion audit check failed:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("Promotion audit check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
