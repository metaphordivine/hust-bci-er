import json
import shutil
from pathlib import Path

import hust_bci_er.training.reproducibility as reproducibility_module
from hust_bci_er.training.toy_adapter import run_toy_route
from scripts.audit_experiment import run_audit
from scripts import toy_experiment_audit


TOY_ROUTE = Path("configs/routes/models/toy_eegnet.yaml")


def test_toy_route_adapter_writes_auditable_smoke(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setattr(reproducibility_module, "PROCESS_START_PYTHONHASHSEED", "42")

    artifacts = run_toy_route(route_config_path=TOY_ROUTE, run_dir=tmp_path / "toy_run")

    assert artifacts.manifest_json.exists()
    assert artifacts.prediction_csv.exists()
    assert artifacts.score_matrix_csv.exists()

    report = run_audit(TOY_ROUTE, artifacts.run_dir, gate="smoke")
    rules = {check["rule_id"]: check["status"] for check in report["checks"]}

    assert rules["MANIFEST_VALID"] == "PASS"
    assert rules["RUN_DATASET_EVIDENCE_VALID"] == "PASS"
    assert rules["RUN_DATASET_LABEL_SCOPE"] == "PASS"
    assert rules["RUN_SPLIT_EVIDENCE_VALID"] == "PASS"
    assert rules["PRIMARY_METRIC_RECOMPUTE"] == "PASS"
    assert rules["PREDICTION_TOP4_RANKING"] == "PASS"


def test_toy_experiment_audit_candidate_fallback_uses_run_local_summary(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setattr(reproducibility_module, "PROCESS_START_PYTHONHASHSEED", "42")

    run_dir = Path("outputs") / "test_toy_script_candidate"
    repo_summary = Path("reports") / "route_summaries" / "toy_eegnet_summary.md"
    shutil.rmtree(run_dir, ignore_errors=True)

    try:
        assert toy_experiment_audit.main(["--run-dir", str(run_dir)]) == 0

        report = json.loads((run_dir / "audit_report.json").read_text(encoding="utf-8"))
        assert report["overall"] == "PASS"
        summary = run_dir / "route_summaries" / "toy_eegnet_summary.md"
        assert summary.exists()
        assert "audit_decision: PASS" in summary.read_text(encoding="utf-8")
        assert not repo_summary.exists()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
