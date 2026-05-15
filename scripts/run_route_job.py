from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.training.toy_adapter import run_toy_route  # noqa: E402


def load_protocol_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute one materialized protocol job with the supported route adapter.")
    parser.add_argument("--protocol-run", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)

    protocol_manifest = load_protocol_manifest(args.protocol_run)
    jobs = {job["job_id"]: job for job in protocol_manifest.get("jobs", []) if isinstance(job, dict)}
    if args.job_id not in jobs:
        raise SystemExit(f"unknown job_id: {args.job_id}")
    job = jobs[args.job_id]
    if "predictions.csv" not in job.get("expected_artifacts", []):
        print(json.dumps({"job_id": args.job_id, "status": "SKIPPED_ARTIFACT_ONLY"}, ensure_ascii=False))
        return 0

    protocol_root = args.protocol_run.resolve().parent
    output_root = (args.output_root or protocol_root / "job_runs").resolve()
    run_dir = output_root / args.job_id
    artifacts = run_toy_route(
        route_config_path=ROOT / job["route_config"],
        run_dir=run_dir,
        split_id=str(job["split_id"]),
        seed=int(job["seed"]),
        command=["python", "scripts/run_route_job.py", "--protocol-run", args.protocol_run.as_posix(), "--job-id", args.job_id],
    )
    print(json.dumps({"job_id": args.job_id, "run_dir": str(artifacts.run_dir), "manifest": str(artifacts.manifest_json)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

