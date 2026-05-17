from __future__ import annotations

import argparse
import json
import math
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


def _required_protocol_path(protocol_root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} is declared but empty")
    root = protocol_root.resolve()
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} must stay under protocol run: {value}") from exc
    if not path.exists():
        raise ValueError(f"{field} declares a missing protocol artifact: {value}")
    return path


def _selection_artifacts(protocol_root: Path, job: dict) -> list[dict[str, str]]:
    from hust_bci_er.audit.manifest import sha256_file

    out: list[dict[str, str]] = []
    for value in job.get("selection_artifact_paths") or []:
        path = _required_protocol_path(protocol_root, value, field="selection_artifact_paths")
        out.append({"path": str(path), "sha256": sha256_file(path)})
    return out


def _selection_metric_mode(metric_name: str) -> str:
    lower = metric_name.lower()
    return "min" if lower == "loss" or lower.endswith("_loss") or "loss" in lower else "max"


def _metric_value(payload: dict, *, path: Path) -> float:
    value = payload.get("metric_value")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"selection metric is not numeric in {path}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"selection metric is not finite in {path}: {value!r}")
    return parsed


def _param_sort_value(value: object) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _select_p3_artifact(protocol_root: Path, job: dict) -> dict | None:
    from hust_bci_er.audit.manifest import sha256_file

    selection_paths = job.get("selection_artifact_paths") or []
    if not selection_paths:
        return None

    loaded: list[dict] = []
    for value in selection_paths:
        path = _required_protocol_path(protocol_root, value, field="selection_artifact_paths")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"selection artifact must be a JSON object: {path}")
        if str(payload.get("stage", "")) != "inner_select":
            raise ValueError(f"selection artifact stage must be inner_select: {path}")
        primary_metric = str(payload.get("primary_metric") or "")
        if not primary_metric:
            raise ValueError(f"selection artifact missing primary_metric: {path}")
        checkpoint_rel = payload.get("checkpoint_path")
        checkpoint_path = _required_protocol_path(path.parent, checkpoint_rel, field="selection checkpoint_path")
        checkpoint_sha256 = sha256_file(checkpoint_path)
        declared_sha = payload.get("checkpoint_sha256")
        if isinstance(declared_sha, str) and declared_sha and declared_sha != checkpoint_sha256:
            raise ValueError(f"selection checkpoint sha256 mismatch: {path}")
        loaded.append(
            {
                "payload": payload,
                "metric_value": _metric_value(payload, path=path),
                "selection_metrics_path": path,
                "selection_metrics_protocol_path": path.relative_to(protocol_root).as_posix(),
                "selection_metrics_sha256": sha256_file(path),
                "checkpoint_path": checkpoint_path,
                "checkpoint_protocol_path": checkpoint_path.relative_to(protocol_root).as_posix(),
                "checkpoint_sha256": checkpoint_sha256,
            }
        )

    primary_metrics = {str(item["payload"].get("primary_metric")) for item in loaded}
    if len(primary_metrics) != 1:
        raise ValueError("selection artifacts disagree on primary_metric: " + ", ".join(sorted(primary_metrics)))
    primary_metric = next(iter(primary_metrics))
    mode = _selection_metric_mode(primary_metric)

    by_param: dict[object, list[dict]] = {}
    for item in loaded:
        by_param.setdefault(item["payload"].get("param_index"), []).append(item)

    group_summaries = []
    for param_index, items in by_param.items():
        values = [float(item["metric_value"]) for item in items]
        group_summaries.append(
            {
                "param_index": param_index,
                "n_inner_artifacts": len(items),
                "mean_metric_value": sum(values) / len(values),
                "metric_values": values,
            }
        )
    group_summaries.sort(
        key=lambda item: (
            -float(item["mean_metric_value"]) if mode == "max" else float(item["mean_metric_value"]),
            _param_sort_value(item["param_index"]),
        )
    )
    selected_param = group_summaries[0]["param_index"]
    selected_group = [item for item in loaded if item["payload"].get("param_index") == selected_param]
    selected_group.sort(
        key=lambda item: (
            -float(item["metric_value"]) if mode == "max" else float(item["metric_value"]),
            _param_sort_value(item["payload"].get("inner_fold")),
            str(item["payload"].get("job_id", "")),
        )
    )
    selected = selected_group[0]
    selected_payload = selected["payload"]
    return {
        "selection_rule": "mean_inner_primary_metric_by_param_index",
        "protocol": str(selected_payload.get("protocol", "")),
        "stage": str(selected_payload.get("stage", "")),
        "primary_metric": primary_metric,
        "metric_mode": mode,
        "selected_param_index": selected_param,
        "selected_metric_mean": group_summaries[0]["mean_metric_value"],
        "selected_metric_value": float(selected["metric_value"]),
        "source_job_id": str(selected_payload.get("job_id", "")),
        "outer_fold": selected_payload.get("outer_fold"),
        "inner_fold": selected_payload.get("inner_fold"),
        "selection_metrics_path": str(selected["selection_metrics_path"]),
        "selection_metrics_protocol_path": selected["selection_metrics_protocol_path"],
        "selection_metrics_sha256": selected["selection_metrics_sha256"],
        "checkpoint_path": str(selected["checkpoint_path"]),
        "checkpoint_protocol_path": selected["checkpoint_protocol_path"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "param_group_summaries": group_summaries,
    }


def _patch_selection_artifact_provenance(run_dir: Path, artifacts: list[dict[str, str]], selected_artifact: dict | None) -> None:
    if not artifacts and selected_artifact is None:
        return
    for name in ("manifest.json", "model_state.local.json"):
        path = run_dir / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if artifacts:
                payload["protocol_selection_artifacts"] = artifacts
            if selected_artifact is not None:
                payload["protocol_selected_artifact"] = selected_artifact
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
    reuse_checkpoint_path = (
        _required_protocol_path(protocol_root, job.get("reuse_checkpoint_path"), field="reuse_checkpoint_path")
        if "reuse_checkpoint_path" in job
        else None
    )
    selection_artifacts = _selection_artifacts(protocol_root, job)
    selected_artifact = _select_p3_artifact(protocol_root, job)
    if reuse_checkpoint_path is None and selected_artifact is not None:
        reuse_checkpoint_path = Path(str(selected_artifact["checkpoint_path"]))
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
            reuse_checkpoint_path=reuse_checkpoint_path,
            reuse_checkpoint_context=selected_artifact,
        )
    _patch_selection_artifact_provenance(run_dir, selection_artifacts, selected_artifact)
    print(json.dumps({"job_id": args.job_id, "run_dir": str(artifacts.run_dir), "manifest": str(artifacts.manifest_json)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

