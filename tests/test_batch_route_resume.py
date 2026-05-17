from __future__ import annotations

import json

import scripts.run_all_torch_routes_candidate as candidate_batch
import scripts.run_all_torch_routes_diagnostic as diagnostic_batch


def _write_torch_route(path, *, route_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"route_id: {route_id}",
                "model: {name: shallow_conv_net}",
                "training: {job_adapter: torch_classifier}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_diagnostic_batch_resume_skips_existing_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnostic_batch, "ROOT", tmp_path)
    route_path = tmp_path / "configs" / "routes" / "models" / "r1.yaml"
    _write_torch_route(route_path, route_id="r1")
    output_dir = tmp_path / "outputs" / "batch_diagnostic"
    output_dir.mkdir(parents=True)
    (output_dir / "batch_summary.json").write_text(
        json.dumps({"results": [{"route_id": "r1", "passed": True, "run_dir": "old"}]}),
        encoding="utf-8",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("existing passed diagnostic route should be skipped")

    monkeypatch.setattr(diagnostic_batch, "run_route_diagnostic", fail_if_called)

    assert diagnostic_batch.main(["--data-root", "data", "--output-dir", str(output_dir)]) == 0

    summary = json.loads((output_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["skipped_existing"] == 1
    assert summary["results"][0]["resume_status"] == "SKIPPED_EXISTING_PASS"


def test_diagnostic_batch_accepts_relative_output_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(diagnostic_batch, "ROOT", tmp_path)
    route_path = tmp_path / "configs" / "routes" / "models" / "r1.yaml"
    _write_torch_route(route_path, route_id="r1")

    def fake_run_route_diagnostic(*args, **kwargs):
        return {"route_id": "r1", "passed": True}

    monkeypatch.setattr(diagnostic_batch, "run_route_diagnostic", fake_run_route_diagnostic)

    assert diagnostic_batch.main(["--data-root", "data", "--output-dir", "relative_out"]) == 0
    assert (tmp_path / "relative_out" / "batch_summary.json").exists()


def test_candidate_batch_resume_skips_existing_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(candidate_batch, "ROOT", tmp_path)
    route_path = tmp_path / "configs" / "routes" / "models" / "r1.yaml"
    _write_torch_route(route_path, route_id="r1")
    output_dir = tmp_path / "outputs" / "batch_candidate"
    output_dir.mkdir(parents=True)
    (output_dir / "candidate_summary.json").write_text(
        json.dumps({"results": [{"route_id": "r1", "passed": True, "run_dir": "old"}]}),
        encoding="utf-8",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("existing passed candidate route should be skipped")

    monkeypatch.setattr(candidate_batch, "run_route_candidate", fail_if_called)

    assert candidate_batch.main(["--route-ids", "r1", "--data-root", "data", "--output-dir", str(output_dir)]) == 0

    summary = json.loads((output_dir / "candidate_summary.json").read_text(encoding="utf-8"))
    assert summary["skipped_existing"] == 1
    assert summary["results"][0]["resume_status"] == "SKIPPED_EXISTING_PASS"


def test_candidate_batch_accepts_relative_output_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(candidate_batch, "ROOT", tmp_path)
    route_path = tmp_path / "configs" / "routes" / "models" / "r1.yaml"
    _write_torch_route(route_path, route_id="r1")

    def fake_run_route_candidate(*args, **kwargs):
        return {"route_id": "r1", "passed": True, "elapsed_sec": 0.0, "returncode": 0}

    monkeypatch.setattr(candidate_batch, "run_route_candidate", fake_run_route_candidate)

    assert candidate_batch.main(["--route-ids", "r1", "--data-root", "data", "--output-dir", "relative_out"]) == 0
    assert (tmp_path / "relative_out" / "candidate_summary.json").exists()
