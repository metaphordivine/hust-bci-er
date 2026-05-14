import csv
import json
from pathlib import Path

import pytest
import yaml

from hust_bci_er.audit.cache_manager import CacheManager
from hust_bci_er.audit.promotion import check_candidate_audit, check_promotion_fields, parse_key_value_markdown
from hust_bci_er.audit.source_scanner import scan_no_leakage
from hust_bci_er.audit.summary import render_route_summary
from hust_bci_er.config.consistency import registry_consistency_errors
from hust_bci_er.config.docs import render_component_docs
from hust_bci_er.contracts.artifacts import ComponentScoreRecord, validate_component_score_rows, write_component_scores
from hust_bci_er.contracts.records import PredictionRecord
from hust_bci_er.evaluation.report import build_metric_report, write_metric_report
from hust_bci_er.evaluation.prediction_writer import write_predictions
from hust_bci_er.training.monitor import TrainingMonitor
from scripts.check_promotion_audit import route_top4_required


def test_component_artifact_and_prediction_writer_contracts(tmp_path):
    component_csv = tmp_path / "component_scores.csv"
    component_meta = write_component_scores(
        [ComponentScoreRecord("c1", "s1", "t1", 0, 0.9, y_true=1)],
        component_csv,
    )
    assert component_meta["rows"] == 1
    assert validate_component_score_rows([{"component_id": "c1", "subject_id": "s1", "trial_id": "t1", "crop_id": 0, "score": 0.9}]) == []
    bad_rows = [
        {"component_id": "c1", "subject_id": "s1", "trial_id": "t1", "crop_id": 0, "score": "nan", "y_true": 3},
        {"component_id": "c1", "subject_id": "s1", "trial_id": "t1", "crop_id": 0, "score": 0.1},
    ]
    bad_errors = validate_component_score_rows(bad_rows)
    assert any("not finite" in err for err in bad_errors)
    assert any("not binary" in err for err in bad_errors)
    assert any("duplicates" in err for err in bad_errors)

    records = [
        PredictionRecord("r1", "s1", f"t{idx}", None, 1.0 - idx * 0.1, y_true=1 if idx < 4 else 0)
        for idx in range(8)
    ]
    prediction_csv = tmp_path / "predictions.csv"
    pred_meta = write_predictions(records, prediction_csv)
    assert pred_meta["rows"] == 8
    text = prediction_csv.read_text(encoding="utf-8")
    assert "pred_top4" in text
    assert text.count(",1,") >= 4

    with pytest.raises(ValueError, match="must contain 8 rows"):
        write_predictions(records[:3], tmp_path / "bad_predictions.csv")

    no_top4_records = [
        PredictionRecord("r1", "s1", "t0", None, 0.2, y_pred=0, y_true=0),
        PredictionRecord("r1", "s1", "t1", None, 0.8, y_pred=1, y_true=1),
        PredictionRecord("r1", "s1", "t2", None, 0.4, y_pred=0, y_true=0),
    ]
    no_top4_csv = tmp_path / "no_top4_predictions.csv"
    no_top4_meta = write_predictions(no_top4_records, no_top4_csv, include_top4=False)
    no_top4_rows = list(csv.DictReader(no_top4_csv.open(newline="", encoding="utf-8")))
    assert no_top4_meta["rows"] == 3
    assert [row["y_pred"] for row in no_top4_rows] == ["0", "1", "0"]
    assert [row["pred_top4"] for row in no_top4_rows] == ["", "", ""]


def test_metric_report_builder_outputs_board_subject_and_audit(tmp_path):
    records = [
        PredictionRecord("r1", "s1", f"t{idx}", None, 1.0 - idx * 0.1, y_true=1 if idx < 4 else 0)
        for idx in range(8)
    ]
    prediction_csv = tmp_path / "predictions.csv"
    write_predictions(records, prediction_csv)

    report = build_metric_report(route_id="r1", prediction_csv=prediction_csv)
    paths = write_metric_report(report, tmp_path / "report")

    assert report["metrics"]["top4_BA"] == 1.0
    assert Path(paths["board"]).exists()
    assert Path(paths["subject_ba"]).exists()
    assert json.loads(Path(paths["audit_json"]).read_text(encoding="utf-8"))["route_id"] == "r1"


def test_metric_report_builder_uses_y_pred_for_no_top4_metric(tmp_path):
    prediction_csv = tmp_path / "predictions.csv"
    prediction_csv.write_text(
        "subject_id,trial_id,y_score,y_pred,y_true,pred_top4\n"
        "s1,t0,0.1,0,0,1\n"
        "s1,t1,0.2,1,1,0\n"
        "s1,t2,0.3,0,0,1\n"
        "s1,t3,0.4,1,1,0\n",
        encoding="utf-8",
    )

    report = build_metric_report(route_id="r1", prediction_csv=prediction_csv, primary_metric="no_top4_BA")

    assert report["metrics"]["no_top4_BA"] == 1.0


def test_metric_report_builder_accepts_prediction_alias_columns(tmp_path):
    no_top4_csv = tmp_path / "no_top4_alias.csv"
    no_top4_csv.write_text(
        "user_id,trial_id,score,pred,y_true\n"
        "s1,t0,0.1,0,0\n"
        "s1,t1,0.8,1,1\n",
        encoding="utf-8",
    )

    no_top4_report = build_metric_report(route_id="r1", prediction_csv=no_top4_csv, primary_metric="no_top4_BA")

    assert no_top4_report["metrics"]["no_top4_BA"] == 1.0

    top4_csv = tmp_path / "top4_alias.csv"
    top4_csv.write_text(
        "subject_id,trial_id,score,y_pred_top4,y_true\n"
        + "\n".join(f"s1,t{idx},{1.0 - idx * 0.01},{1 if idx < 4 else 0},{1 if idx < 4 else 0}" for idx in range(8))
        + "\n",
        encoding="utf-8",
    )

    top4_report = build_metric_report(route_id="r1", prediction_csv=top4_csv, primary_metric="top4_BA")

    assert top4_report["metrics"]["top4_BA"] == 1.0


def test_prediction_report_all_correct_rate_uses_group_all_or_none_and_float_binary(tmp_path):
    prediction_csv = tmp_path / "all_correct_predictions.csv"
    lines = ["subject_id,trial_id,y_pred,y_true"]
    for idx in range(8):
        y_true = 1.0 if idx < 4 else 0.0
        y_pred = 0.0 if idx == 0 else y_true
        lines.append(f"s1,t{idx},{y_pred},{y_true}")
    prediction_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = build_metric_report(route_id="r1", prediction_csv=prediction_csv, primary_metric="all_correct_rate")

    assert report["metrics"]["all_correct_rate"] == 0.0
    assert report["subjects"][0]["metric_value"] == 0.0


def test_metric_report_builder_supports_score_matrix_exact_metric(tmp_path):
    score_matrix = tmp_path / "score_matrix.csv"
    lines = ["subject_id,trial_id,y_true,crop_0,crop_1,crop_2,crop_3,crop_4"]
    for idx in range(8):
        label = 1 if idx < 4 else 0
        score = 1.0 - idx * 0.01
        lines.append(f"s1,t{idx},{label},{score},{score},{score},{score},{score}")
    score_matrix.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = build_metric_report(route_id="r1", score_matrix_csv=score_matrix, primary_metric="exact_single_crop_expected_BA")
    paths = write_metric_report(report, tmp_path / "score_report")

    assert report["metrics"]["exact_single_crop_expected_BA"] == 1.0
    assert report["score_matrix_sha256"]
    assert "metric_value" in Path(paths["subject_ba"]).read_text(encoding="utf-8").splitlines()[0]


def test_score_matrix_report_preserves_subject_with_composite_group_keys(tmp_path):
    score_matrix = tmp_path / "score_matrix.csv"
    lines = ["seed,fold,subject_id,trial_id,y_true,crop_0,crop_1,crop_2,crop_3,crop_4"]
    for idx in range(8):
        label = 1 if idx < 4 else 0
        score = 1.0 - idx * 0.01
        lines.append(f"42,0,s1,t{idx},{label},{score},{score},{score},{score},{score}")
    score_matrix.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = build_metric_report(route_id="r1", score_matrix_csv=score_matrix, primary_metric="exact_single_crop_expected_BA", metric_group_keys=("seed", "fold", "subject_id"))

    assert report["subjects"][0]["subject_id"] == "s1"
    assert report["subjects"][0]["group_key"] == "42|0|s1"


def test_prediction_report_respects_composite_metric_group_keys(tmp_path):
    prediction_csv = tmp_path / "predictions.csv"
    prediction_csv.write_text(
        "seed,fold,subject_id,trial_id,y_pred,y_true\n"
        "42,0,s1,t0,0,0\n"
        "42,0,s1,t1,1,1\n"
        "43,0,s1,t0,1,0\n"
        "43,0,s1,t1,0,1\n",
        encoding="utf-8",
    )

    report = build_metric_report(route_id="r1", prediction_csv=prediction_csv, primary_metric="no_top4_BA", metric_group_keys=("seed", "fold", "subject_id"))

    assert [row["group_key"] for row in report["subjects"]] == ["42|0|s1", "43|0|s1"]
    assert report["metrics"]["no_top4_BA"] == 0.5


def test_source_scanner_finds_leakage_pattern(tmp_path):
    root = tmp_path
    src = root / "src"
    src.mkdir()
    (src / "bad.py").write_text("value = 'public_label'\n", encoding="utf-8")

    findings = scan_no_leakage(root)

    assert len(findings) == 1
    assert findings[0].pattern == "public_label"


def test_source_scanner_finds_id_feature_and_inference_label_leaks(tmp_path):
    root = tmp_path
    features = root / "src" / "hust_bci_er" / "features"
    inference = root / "src" / "hust_bci_er" / "inference"
    scripts = root / "scripts"
    features.mkdir(parents=True)
    inference.mkdir(parents=True)
    scripts.mkdir()
    (features / "bad_feature.py").write_text("features.append(row['subject_id'])\n", encoding="utf-8")
    (inference / "bad_infer.py").write_text("return batch['y_true']\n", encoding="utf-8")
    (scripts / "infer.py").write_text("return batch['y_true']\n", encoding="utf-8")

    patterns = {finding.pattern for finding in scan_no_leakage(root)}

    assert "id_shortcut_as_feature" in patterns
    assert sum(1 for finding in scan_no_leakage(root) if finding.pattern == "inference_label_reference") == 2


def test_source_scanner_allows_score_route_assembly_audit_label_passthrough(tmp_path):
    root = tmp_path
    inference = root / "src" / "hust_bci_er" / "inference"
    inference.mkdir(parents=True)
    (inference / "score_route_assembly.py").write_text("row['y_true'] = label\n", encoding="utf-8")
    (inference / "bad_infer.py").write_text("return row['y_true']\n", encoding="utf-8")

    findings = scan_no_leakage(root)

    assert all(finding.path != "src/hust_bci_er/inference/score_route_assembly.py" for finding in findings)
    assert any(finding.path == "src/hust_bci_er/inference/bad_infer.py" and finding.pattern == "inference_label_reference" for finding in findings)


def test_summary_and_promotion_helpers(tmp_path):
    summary = render_route_summary(
        route_data={"route_id": "r1", "status": "IDEA", "dataset_version": "d1", "split_id": "s1", "seed": 42, "evaluation": {"protocol": "p1", "primary_metric": "top4_BA"}},
        audit_report={"route_id": "r1", "overall": "PASS", "gate": "candidate", "route_config": "configs/routes/models/r1.yaml", "run_dir": "outputs/r1/run", "metrics": {"top4_BA": 0.75}},
    )
    assert "route_id: r1" in summary
    assert "primary_metric_value: 0.75" in summary

    promotion = tmp_path / "promotion.md"
    promotion.write_text(
        "\n".join(
            [
                "route_id: r1",
                "promoted_from_run: outputs/r1/run",
                "candidate_audit_report: candidate.json",
                "primary_metric: top4_BA",
                "comparison_baseline: baseline",
                "risk_review: reviewed",
                "no_leakage_review: reviewed",
                "decision: promote",
                "reviewer: test",
                "date: 2026-05-14",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fields = parse_key_value_markdown(promotion)
    assert check_promotion_fields(fields) == []
    candidate = tmp_path / "candidate.json"
    critical_rules = [
        "MANIFEST_VALID",
        "PRIMARY_METRIC_RECOMPUTE",
        "PRIMARY_METRIC_REPORTED",
        "RUN_DATASET_EVIDENCE_VALID",
        "RUN_SPLIT_EVIDENCE_VALID",
        "RUN_SPLIT_EVIDENCE_CONSISTENT",
        "RUN_REPRODUCIBILITY_LOCKED",
    ]
    candidate.write_text(
        json.dumps({"route_id": "r1", "gate": "candidate", "overall": "PASS", "checks": [{"rule_id": rule, "status": "PASS"} for rule in critical_rules]}),
        encoding="utf-8",
    )
    assert check_candidate_audit(candidate, route_id="r1") == []

    no_top4_candidate = tmp_path / "no_top4_candidate.json"
    no_top4_candidate.write_text(
        json.dumps({"route_id": "r1", "gate": "candidate", "overall": "PASS", "checks": [{"rule_id": rule, "status": "PASS"} for rule in critical_rules + ["PREDICTION_TOP4_GROUPS"]]}),
        encoding="utf-8",
    )
    assert check_candidate_audit(no_top4_candidate, route_id="r1") == []

    top4_missing_truth_balance = tmp_path / "top4_missing_truth_balance.json"
    top4_checks = [{"rule_id": rule, "status": "PASS"} for rule in critical_rules + ["PREDICTION_TOP4_RANKING", "PREDICTION_TOP4_BINARY", "PREDICTION_TRIAL_ID_UNIQUE"]]
    top4_missing_truth_balance.write_text(
        json.dumps({"route_id": "r1", "gate": "candidate", "overall": "PASS", "checks": top4_checks}),
        encoding="utf-8",
    )
    assert check_candidate_audit(top4_missing_truth_balance, route_id="r1") == []
    assert "PREDICTION_TOP4_TRUTH_BALANCE" in "; ".join(check_candidate_audit(top4_missing_truth_balance, route_id="r1", top4_required=True))

    minimal = tmp_path / "minimal_candidate.json"
    minimal.write_text(json.dumps({"route_id": "r1", "gate": "candidate", "overall": "PASS"}), encoding="utf-8")
    assert "checks list" in "; ".join(check_candidate_audit(minimal, route_id="r1"))


def test_route_summary_normalizes_repo_relative_evidence_paths(tmp_path):
    root = tmp_path
    run_dir = root / "outputs" / "r1" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "audit_report.json").write_text(json.dumps({"route_id": "r1", "gate": "candidate", "overall": "PASS"}), encoding="utf-8")
    manifest = run_dir / "manifest.json"
    manifest.write_text(json.dumps({"route_id": "r1", "primary_metric": "top4_BA", "metrics": {"top4_BA": 0.5}}), encoding="utf-8")

    summary = render_route_summary(
        route_data={"route_id": "r1", "status": "CANDIDATE", "dataset_version": "d1", "split_id": "s1", "seed": 42, "evaluation": {"protocol": "p1", "primary_metric": "top4_BA"}},
        audit_report={"route_id": "r1", "overall": "PASS", "gate": "candidate", "route_config": "configs/routes/models/r1.yaml", "run_dir": str(run_dir), "metrics": {"top4_BA": 0.5}},
        manifest_path=manifest,
        root=root,
    )

    assert "audit_report_path: outputs/r1/run/audit_report.json" in summary
    assert "manifest_path: outputs/r1/run/manifest.json" in summary
    assert str(root) not in summary
    outside_manifest = tmp_path.parent / "outside_manifest.json"
    outside_manifest.write_text(json.dumps({"route_id": "r1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_path must be inside repository"):
        render_route_summary(
            route_data={"route_id": "r1", "evaluation": {"primary_metric": "top4_BA"}},
            audit_report={"route_id": "r1", "overall": "PASS", "gate": "candidate", "run_dir": str(run_dir)},
            manifest_path=outside_manifest,
            root=root,
        )
    with pytest.raises(ValueError, match="manifest_path must be inside repository"):
        render_route_summary(
            route_data={"route_id": "r1", "evaluation": {"primary_metric": "top4_BA"}},
            audit_report={"route_id": "r1", "overall": "PASS", "gate": "candidate", "run_dir": str(run_dir)},
            manifest_path=tmp_path.parent / "missing_outside_manifest.json",
            root=root,
        )
    with pytest.raises(ValueError, match="manifest_path does not exist"):
        render_route_summary(
            route_data={"route_id": "r1", "evaluation": {"primary_metric": "top4_BA"}},
            audit_report={"route_id": "r1", "overall": "PASS", "gate": "candidate", "run_dir": str(run_dir)},
            manifest_path=root / "missing_manifest.json",
            root=root,
        )


def test_promotion_cli_derives_top4_requirement_from_route_config(monkeypatch, tmp_path):
    from scripts import check_promotion_audit as promotion_cli

    monkeypatch.setattr(promotion_cli, "ROOT", tmp_path)
    route = tmp_path / "r1.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: r1",
                "inference:",
                "  top4: true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert route_top4_required("r1", route) is True
    candidate = tmp_path / "candidate.json"
    critical_rules = [
        "MANIFEST_VALID",
        "PRIMARY_METRIC_RECOMPUTE",
        "PRIMARY_METRIC_REPORTED",
        "RUN_DATASET_EVIDENCE_VALID",
        "RUN_SPLIT_EVIDENCE_VALID",
        "RUN_SPLIT_EVIDENCE_CONSISTENT",
        "RUN_REPRODUCIBILITY_LOCKED",
    ]
    candidate.write_text(
        json.dumps({"route_id": "r1", "gate": "candidate", "overall": "PASS", "checks": [{"rule_id": rule, "status": "PASS"} for rule in critical_rules]}),
        encoding="utf-8",
    )
    promotion = tmp_path / "promotion.md"
    promotion.write_text(
        "\n".join(
            [
                "route_id: r1",
                "promoted_from_run: outputs/r1/run",
                "candidate_audit_report: candidate.json",
                "primary_metric: top4_BA",
                "comparison_baseline: baseline",
                "risk_review: reviewed",
                "no_leakage_review: reviewed",
                "decision: promote",
                "reviewer: test",
                "date: 2026-05-14",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert promotion_cli.main(["--promotion-audit", str(promotion), "--route-config", str(route)]) == 1


def test_promotion_cli_rejects_candidate_report_outside_repo(monkeypatch, tmp_path, capsys):
    from scripts import check_promotion_audit as promotion_cli

    monkeypatch.setattr(promotion_cli, "ROOT", tmp_path)
    route = tmp_path / "r1.yaml"
    route.write_text("route_id: r1\ninference:\n  top4: false\n", encoding="utf-8")
    promotion = tmp_path / "promotion.md"
    promotion.write_text(
        "\n".join(
            [
                "route_id: r1",
                "promoted_from_run: outputs/r1/run",
                "candidate_audit_report: ../outside.json",
                "primary_metric: top4_BA",
                "comparison_baseline: baseline",
                "risk_review: reviewed",
                "no_leakage_review: reviewed",
                "decision: promote",
                "reviewer: test",
                "date: 2026-05-14",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert promotion_cli.main(["--promotion-audit", str(promotion), "--route-config", str(route)]) == 1
    assert "candidate_audit_report must be inside repository" in capsys.readouterr().out


def test_registry_docs_checker_monitor_and_cache_manager(tmp_path):
    assert registry_consistency_errors(Path.cwd()) == []
    docs = render_component_docs()
    assert "Torch Backbones" in docs
    assert "deformer_lite" in docs

    monitor = TrainingMonitor(tmp_path / "run", route_id="r1", run_id="run")
    monitor.epoch(1, {"loss": 0.5})
    monitor.finish()
    assert (tmp_path / "run" / "training_events.jsonl").exists()
    assert json.loads((tmp_path / "run" / "heartbeat.json").read_text(encoding="utf-8"))["event"] == "finish"

    cache_dir = tmp_path / "cache"
    artifact = cache_dir / "features.npy"
    cache_dir.mkdir()
    artifact.write_bytes(b"features")
    entry = CacheManager(cache_dir).register(key="features", path=artifact, kind="feature")
    assert entry["path"] == "features.npy"
    with pytest.raises(ValueError):
        CacheManager(cache_dir).register(key="bad", path=tmp_path / "outside.bin", kind="feature")
