from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.data.manifest_builder import build_dataset_manifest, read_index_csv, write_dataset_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an auditable dataset manifest from an index CSV.")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--allow-missing-files", action="store_true")
    args = parser.parse_args(argv)

    rows = read_index_csv(args.index_csv)
    manifest = build_dataset_manifest(
        dataset_version=args.dataset_version,
        rows=rows,
        base_dir=args.base_dir or args.index_csv.parent,
        require_files=not args.allow_missing_files,
    )
    write_dataset_manifest(manifest, args.output)
    print(f"Wrote dataset manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
