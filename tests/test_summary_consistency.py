import json
from pathlib import Path

from hust_bci_er.audit.manifest import sha256_file
from scripts import check_summary_consistency


def write_route(root: Path, *, status: str = "CANDIDATE") -> None:
    route_dir = root / "configs" / "routes" / "models"
    route_dir.mkdir(parents=True)
    (route_dir / "summary_route.yaml").write_text(
        "\n".join(
            [
                "route_id: summary_route",
                f"status: {status}",
                "dataset_version: train_v1",
                "split_id: p1_seed42_fold0",
                "seed: 42",
                "input_window_sec: 10",
                "model: {name: deformer_lite}",
                "inference: {top4: true, crop_policy: single}",
                "evaluation: {protocol: p1_repeated_group_kfold, primary_metric: top4_BA}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_bound_summary(
    root: Path,
    *,
    metric_value: str = "0.75",
    manifest_hash: str | None = None,
    manifest_payload: dict | None = None,
    manifest_path: str = "reports/audits/manifest.json",
    reproduce: str = "test",
) -> None:
    reports = root / "reports" / "route_summaries"
    reports.mkdir(parents=True)
    audit = root / "reports" / "audits" / "audit_report.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps({"route_id": "summary_route", "gate": "candidate", "overall": "PASS"}), encoding="utf-8")
    manifest = root / manifest_path
    manifest.mkdir(parents=True, exist_ok=True) if manifest_path.endswith("/") else manifest.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_path.endswith("/"):
        manifest.write_text(
            json.dumps(manifest_payload or {"route_id": "summary_route", "primary_metric": "top4_BA", "metrics": {"top4_BA": 0.75}}),
            encoding="utf-8",
        )
    manifest_hash = manifest_hash or sha256_file(manifest)
    (reports / "summary_route_summary.md").write_text(
        "\n".join(
            [
                "route_id: summary_route",
                "route_status: CANDIDATE",
                "audit_decision: PASS",
                "gate: candidate",
                "primary_metric: top4_BA",
                f"primary_metric_value: {metric_value}",
                "audit_report_path: reports/audits/audit_report.json",
                f"manifest_path: {manifest_path}",
                f"manifest_sha256: {manifest_hash}",
                "decision: keep",
                f"reproduce: {reproduce}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_advanced_summary_accepts_bound_manifest(monkeypatch, tmp_path):
    write_route(tmp_path)
    write_bound_summary(tmp_path)
    monkeypatch.setattr(check_summary_consistency, "ROOT", tmp_path)

    assert check_summary_consistency.main() == 0


def test_advanced_summary_rejects_manifest_hash_and_metric_mismatch(monkeypatch, tmp_path, capsys):
    write_route(tmp_path)
    write_bound_summary(tmp_path, metric_value="0.50", manifest_hash="0" * 64)
    monkeypatch.setattr(check_summary_consistency, "ROOT", tmp_path)

    assert check_summary_consistency.main() == 1
    output = capsys.readouterr().out
    assert "summary manifest_sha256 mismatch" in output
    assert "summary primary_metric_value mismatch" in output


def test_advanced_summary_rejects_directory_manifest_path(monkeypatch, tmp_path, capsys):
    write_route(tmp_path)
    manifest_dir = tmp_path / "reports" / "audits" / "manifest_dir"
    manifest_dir.mkdir(parents=True)
    write_bound_summary(tmp_path, manifest_path="reports/audits/manifest_dir/", manifest_hash="0" * 64)
    monkeypatch.setattr(check_summary_consistency, "ROOT", tmp_path)

    assert check_summary_consistency.main() == 1
    assert "summary manifest_path is not a file" in capsys.readouterr().out


def test_advanced_summary_rejects_non_finite_metric_values(monkeypatch, tmp_path, capsys):
    write_route(tmp_path)
    write_bound_summary(
        tmp_path,
        metric_value="NaN",
        manifest_payload={"route_id": "summary_route", "primary_metric": "top4_BA", "metrics": {"top4_BA": 0.75}},
    )
    monkeypatch.setattr(check_summary_consistency, "ROOT", tmp_path)

    assert check_summary_consistency.main() == 1
    assert "summary primary_metric_value is not numeric" in capsys.readouterr().out


def test_advanced_summary_rejects_missing_manifest_primary_metric(monkeypatch, tmp_path, capsys):
    write_route(tmp_path)
    write_bound_summary(
        tmp_path,
        manifest_payload={"route_id": "summary_route", "metrics": {"top4_BA": 0.75}},
    )
    monkeypatch.setattr(check_summary_consistency, "ROOT", tmp_path)

    assert check_summary_consistency.main() == 1
    assert "summary manifest primary_metric missing or invalid" in capsys.readouterr().out


def test_summary_rejects_local_absolute_reproduce_path(monkeypatch, tmp_path, capsys):
    write_route(tmp_path)
    write_bound_summary(
        tmp_path,
        reproduce="python scripts/repo_doctor.py experiment --route D:/repo/configs/routes/models/r.yaml --run outputs/r/run --gate candidate",
    )
    monkeypatch.setattr(check_summary_consistency, "ROOT", tmp_path)

    assert check_summary_consistency.main() == 1
    assert "summary reproduce command contains local absolute path" in capsys.readouterr().out
