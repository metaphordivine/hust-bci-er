from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.audit.manifest import sha256_file  # noqa: E402
from hust_bci_er.data.split_builder import build_split_manifest, load_dataset_manifest, write_split_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic subject-level split manifest.")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--val-count", type=int, required=True)
    parser.add_argument("--test-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    dataset = load_dataset_manifest(args.dataset_manifest)
    manifest = build_split_manifest(
        split_id=args.split_id,
        dataset=dataset,
        seed=args.seed,
        val_count=args.val_count,
        test_count=args.test_count,
        dataset_manifest_path=args.dataset_manifest.as_posix(),
        dataset_manifest_sha256=sha256_file(args.dataset_manifest),
    )
    write_split_manifest(manifest, args.output)
    print(f"Wrote split manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
