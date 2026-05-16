"""Execute one materialized artifact-only protocol job (P2 train_holdout, P3 inner_select).

These protocol stages produce artifacts (checkpoints, selection metrics)
rather than prediction CSVs.  This adapter provides the minimal execution
path for diagnostic/smoke evaluation; full candidate-grade P2/P3 evidence
still requires the formal training/selection adapter.

Usage (called from protocol runner, not directly)::

    python scripts/run_artifact_job.py \
        --protocol-run <path>/protocol_run_manifest.json \
        --job-id p2__route__train_seed42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def load_protocol_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_route(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"route config must be a mapping: {path}")
    return data


def _run_train_holdout(
    job: dict,
    *,
    route_path: Path,
    run_dir: Path,
    split_path: Path,
    device: str,
    data_root: Path | None,
    epochs_override: int | None,
    protocol_manifest: dict,
) -> int:
    """P2 train_holdout_model: train once and export checkpoint."""
    from hust_bci_er.training.real_adapter import run_real_classifier_route  # noqa: E402

    artifacts = run_real_classifier_route(
        route_config_path=route_path,
        run_dir=run_dir,
        run_mode="full_subjects",
        split_id=str(job["split_id"]),
        split_manifest_path=split_path,
        seed=int(job["seed"]),
        command=[
            "python",
            "scripts/run_artifact_job.py",
            "--protocol-run",
            str(protocol_manifest.get("_protocol_run_manifest_path", "")),
            "--job-id",
            str(job["job_id"]),
        ],
        data_root=data_root,
        smoke_epochs=epochs_override,
        device=device,
    )
    print(json.dumps({"job_id": job["job_id"], "run_dir": str(artifacts.run_dir), "manifest": str(artifacts.manifest_json)}, ensure_ascii=False))
    return 0


def _run_inner_select(
    job: dict,
    *,
    route_path: Path,
    run_dir: Path,
    split_path: Path,
    device: str,
    data_root: Path | None,
    epochs_override: int | None,
    protocol_manifest: dict,
) -> int:
    """P3 inner_select: train one config point and record selection metrics.

    Full grid-search inner selection requires the formal P3 adapter.
    This minimal path trains a single config and writes a placeholder
    selection_metrics.json for smoke validation.
    """
    from hust_bci_er.training.real_adapter import run_real_classifier_route  # noqa: E402

    artifacts = run_real_classifier_route(
        route_config_path=route_path,
        run_dir=run_dir,
        run_mode="full_subjects",
        split_id=str(job["split_id"]),
        split_manifest_path=split_path,
        seed=int(job["seed"]),
        command=[
            "python",
            "scripts/run_artifact_job.py",
            "--protocol-run",
            str(protocol_manifest.get("_protocol_run_manifest_path", "")),
            "--job-id",
            str(job["job_id"]),
        ],
        data_root=data_root,
        smoke_epochs=epochs_override,
        device=device,
    )

    selection_metrics = {
        "job_id": str(job["job_id"]),
        "route_id": str(job["route_id"]),
        "param_index": job.get("param_index"),
        "outer_fold": job.get("outer_fold"),
        "inner_fold": job.get("inner_fold"),
        "primary_metric": str(artifacts.metric_report.get("primary_metric", "")),
        "metric_value": artifacts.metric_report.get("metrics", {}).get(
            str(artifacts.metric_report.get("primary_metric", "")),
        ),
        "note": "minimal single-config inner_select; full P3 grid search requires formal adapter",
    }
    (run_dir / "selection_metrics.json").write_text(json.dumps(selection_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"job_id": job["job_id"], "run_dir": str(artifacts.run_dir), "manifest": str(artifacts.manifest_json)}, ensure_ascii=False))
    return 0


ARTIFACT_HANDLERS = {
    "train_holdout_model": _run_train_holdout,
    "inner_select": _run_inner_select,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute an artifact-only protocol job (P2/P3).")
    parser.add_argument("--protocol-run", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--device")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--epochs-override", type=int)
    args = parser.parse_args(argv)

    protocol_manifest = load_protocol_manifest(args.protocol_run)
    # Store the manifest path so handlers can reconstruct it
    protocol_manifest["_protocol_run_manifest_path"] = str(args.protocol_run.resolve())

    effective_device = str(args.device or protocol_manifest.get("default_execute_device") or "auto")
    effective_data_root = args.data_root
    if effective_data_root is None and protocol_manifest.get("data_root"):
        effective_data_root = Path(str(protocol_manifest["data_root"]))

    jobs = {job["job_id"]: job for job in protocol_manifest.get("jobs", []) if isinstance(job, dict)}
    if args.job_id not in jobs:
        raise SystemExit(f"unknown job_id: {args.job_id}")
    job = jobs[args.job_id]

    stage = str(job.get("stage", ""))
    if stage not in ARTIFACT_HANDLERS:
        supported = ", ".join(sorted(ARTIFACT_HANDLERS))
        print(json.dumps(
            {"job_id": args.job_id, "status": "UNSUPPORTED_STAGE", "reason": f"artifact stage {stage} not supported; supported: {supported}"},
            ensure_ascii=False,
        ))
        return 1

    protocol_root = args.protocol_run.resolve().parent
    run_dir = protocol_root / "job_runs" / args.job_id
    route_path = ROOT / job["route_config"]
    split_path = protocol_root / str(job["split_manifest_path"])

    handler = ARTIFACT_HANDLERS[stage]
    return handler(
        job,
        route_path=route_path,
        run_dir=run_dir,
        split_path=split_path,
        device=effective_device,
        data_root=effective_data_root,
        epochs_override=args.epochs_override,
        protocol_manifest=protocol_manifest,
    )


if __name__ == "__main__":
    raise SystemExit(main())
