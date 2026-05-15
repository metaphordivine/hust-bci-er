from pathlib import Path

import hust_bci_er.training.reproducibility as reproducibility_module
from hust_bci_er.training.toy_adapter import run_toy_route
from scripts.audit_experiment import run_audit


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

