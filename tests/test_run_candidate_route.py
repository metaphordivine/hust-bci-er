from __future__ import annotations

import json

import scripts.run_candidate_route as run_candidate_route


def test_candidate_summary_placeholder_is_overwritten_with_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(run_candidate_route, "ROOT", tmp_path)
    route = tmp_path / "route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: r1",
                "status: IDEA",
                "dataset_version: train_v1",
                "split_id: split1",
                "seed: 42",
                "evaluation:",
                "  protocol: p1_repeated_group_kfold",
                "  primary_metric: exact_single_crop_expected_BA",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "audit_report.json").write_text(
        json.dumps(
            {
                "route_id": "r1",
                "route_config": str(route),
                "run_dir": str(run_dir),
                "gate": "candidate",
                "overall": "PASS",
                "checks": [],
                "metrics": {"exact_single_crop_expected_BA": 0.5},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "route_id": "r1",
                "primary_metric": "exact_single_crop_expected_BA",
                "metrics": {"exact_single_crop_expected_BA": 0.5},
            }
        ),
        encoding="utf-8",
    )
    summary_dir = tmp_path / "reports" / "route_summaries"

    placeholder = run_candidate_route._write_summary_placeholder(route, run_dir, summary_dir)
    assert "audit_decision: PENDING" in placeholder.read_text(encoding="utf-8")

    final = run_candidate_route._write_summary(route, run_dir, summary_dir)
    text = final.read_text(encoding="utf-8")
    assert "audit_decision: PASS" in text
    assert "audit_decision: PENDING" not in text


def test_candidate_placeholder_defaults_to_run_local_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(run_candidate_route, "ROOT", tmp_path)
    route = tmp_path / "route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: r1",
                "status: IDEA",
                "dataset_version: train_v1",
                "split_id: split1",
                "seed: 42",
                "evaluation:",
                "  protocol: p1_repeated_group_kfold",
                "  primary_metric: exact_single_crop_expected_BA",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    placeholder = run_candidate_route._write_run_local_summary_placeholder(route, run_dir)
    assert placeholder == run_dir / "route_summaries" / "r1_summary.md"
    assert not (tmp_path / "reports" / "route_summaries" / "r1_summary.md").exists()
