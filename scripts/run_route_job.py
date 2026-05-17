from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.training.toy_adapter import run_toy_route  # noqa: E402


SUPPORTED_ADAPTERS = {"toy_centroid", "torch_classifier"}


def load_protocol_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_route(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"route config must be a mapping: {path}")
    return data


def route_adapter_name(route_data: dict) -> str:
    training = route_data.get("training")
    if not isinstance(training, dict):
        raise ValueError("route config has no training mapping")
    adapter = training.get("job_adapter") or training.get("trainer")
    if adapter not in SUPPORTED_ADAPTERS:
        raise ValueError(f"unsupported protocol job adapter: {adapter}")
    return str(adapter)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute one materialized protocol job with the supported route adapter.")
    parser.add_argument("--protocol-run", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--mode", choices=["smoke", "full_subjects", "candidate"], default="candidate")
    parser.add_argument("--device")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--epochs-override", type=int, help="Diagnostic-only epoch override for protocol job smoke execution.")
    args = parser.parse_args(argv)

    protocol_manifest = load_protocol_manifest(args.protocol_run)
    effective_device = str(args.device or protocol_manifest.get("default_execute_device") or "auto")
    effective_data_root = args.data_root
    if effective_data_root is None and protocol_manifest.get("data_root"):
        effective_data_root = Path(str(protocol_manifest["data_root"]))
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
    route_path = ROOT / job["route_config"]
    route_data = load_route(route_path)
    adapter_name = route_adapter_name(route_data)
    split_path = protocol_root / str(job["split_manifest_path"])
    command = [
        "python",
        "scripts/run_route_job.py",
        "--protocol-run",
        args.protocol_run.as_posix(),
        "--job-id",
        args.job_id,
        "--mode",
        args.mode,
        "--device",
        effective_device,
    ]
    if effective_data_root is not None:
        command.extend(["--data-root", effective_data_root.as_posix()])
    if args.epochs_override is not None:
        command.extend(["--epochs-override", str(args.epochs_override)])

    if adapter_name == "toy_centroid":
        artifacts = run_toy_route(
            route_config_path=route_path,
            run_dir=run_dir,
            split_id=str(job["split_id"]),
            seed=int(job["seed"]),
            command=command,
        )
    else:
        from hust_bci_er.training.real_adapter import run_real_classifier_route  # noqa: E402

        artifacts = run_real_classifier_route(
            route_config_path=route_path,
            run_dir=run_dir,
            run_mode=args.mode,
            split_id=str(job["split_id"]),
            split_manifest_path=split_path,
            seed=int(job["seed"]),
            command=command,
            data_root=effective_data_root,
            smoke_epochs=args.epochs_override,
            device=effective_device,
            crop_policy=job.get("crop_policy") if isinstance(job.get("crop_policy"), dict) else None,
        )
    print(json.dumps({"job_id": args.job_id, "run_dir": str(artifacts.run_dir), "manifest": str(artifacts.manifest_json)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

