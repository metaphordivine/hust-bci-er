from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.audit.promotion import check_candidate_audit, check_promotion_fields, parse_key_value_markdown  # noqa: E402


def route_top4_required(route_id: str, route_config: Path | None = None) -> bool:
    path = route_config or ROOT / "configs" / "routes" / "models" / f"{route_id}.yaml"
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise ValueError(f"route config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"route config must be a mapping: {path}")
    if data.get("route_id") != route_id:
        raise ValueError("route config route_id does not match promotion audit")
    inference = data.get("inference")
    return bool(inference.get("top4")) if isinstance(inference, dict) else False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check promotion audit fields and candidate report binding.")
    parser.add_argument("--promotion-audit", type=Path, required=True)
    parser.add_argument("--route-config", type=Path)
    args = parser.parse_args(argv)
    fields = parse_key_value_markdown(args.promotion_audit)
    errors = check_promotion_fields(fields)
    if fields.get("candidate_audit_report") and fields.get("route_id"):
        try:
            top4_required = route_top4_required(fields["route_id"], args.route_config)
        except ValueError as exc:
            errors.append(str(exc))
            top4_required = False
        errors.extend(check_candidate_audit(ROOT / fields["candidate_audit_report"], route_id=fields["route_id"], top4_required=top4_required))
    if errors:
        print("Promotion audit check failed:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("Promotion audit check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
