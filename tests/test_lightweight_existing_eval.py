from __future__ import annotations

import json
from pathlib import Path

from scripts import lightweight_existing_eval


def _write_route(path: Path, route_id: str, model: str, status: str = "IDEA") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"route_id: {route_id}",
                f"status: {status}",
                f"model: {{name: {model}}}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_lightweight_existing_eval_ranks_protocol_boards(monkeypatch, tmp_path):
    monkeypatch.setattr(lightweight_existing_eval, "ROOT", tmp_path)
    _write_route(tmp_path / "configs" / "routes" / "models" / "r_fbst.yaml", "r_fbst", "fbstcnet")
    _write_route(tmp_path / "configs" / "routes" / "models" / "r_graph.yaml", "r_graph", "dgcnn")
    board = tmp_path / "outputs" / "run" / "protocol3_board.csv"
    board.parent.mkdir(parents=True)
    board.write_text(
        "\n".join(
            [
                "route_id,n_jobs,mean,std,min,max",
                "r_fbst,5,0.61,0.01,0.60,0.62",
                "r_graph,5,0.65,0.02,0.62,0.68",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (board.parent / "protocol3_audit.json").write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")

    report = lightweight_existing_eval.build_report(
        summary_dir=tmp_path / "reports" / "route_summaries",
        run_roots=[tmp_path / "outputs"],
        board_paths=[],
        top_k=5,
    )

    assert "P3: 2 routes with any evidence (2 protocol-board, 0 route-summary-only)" in report
    assert report.index("`r_graph`") < report.index("`r_fbst`")
    assert "graph" in report


def test_lightweight_existing_eval_uses_route_summary_protocol_means(monkeypatch, tmp_path):
    monkeypatch.setattr(lightweight_existing_eval, "ROOT", tmp_path)
    _write_route(tmp_path / "configs" / "routes" / "models" / "r_cb.yaml", "r_cb", "cbramod")
    summary = tmp_path / "reports" / "route_summaries" / "r_cb_summary.md"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        "\n".join(
            [
                "route_id: r_cb",
                "route_status: IDEA",
                "audit_decision: PASS",
                "gate: candidate",
                "primary_metric: exact_single_crop_expected_BA",
                "primary_metric_value: 0.62",
                "aggregate_metric_mean: 0.62",
                "p3_metric_mean: 0.66",
                "decision: hold",
                "reproduce: test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = lightweight_existing_eval.build_report(
        summary_dir=summary.parent,
        run_roots=[],
        board_paths=[],
        top_k=5,
    )

    assert "P1: 1 routes with any evidence (0 protocol-board, 1 route-summary-only)" in report
    assert "P3: 1 routes with any evidence (0 protocol-board, 1 route-summary-only)" in report
    assert "+0.0400" in report


def test_lightweight_existing_eval_skips_incomplete_run_root_boards(monkeypatch, tmp_path):
    monkeypatch.setattr(lightweight_existing_eval, "ROOT", tmp_path)
    _write_route(tmp_path / "configs" / "routes" / "models" / "r_graph.yaml", "r_graph", "dgcnn")
    board = tmp_path / "outputs" / "run" / "protocol2_board.csv"
    board.parent.mkdir(parents=True)
    board.write_text("route_id,n_jobs,mean,std,min,max\nr_graph,2,0.91,,,\n", encoding="utf-8")
    (board.parent / "protocol2_audit.json").write_text(json.dumps({"status": "INCOMPLETE"}), encoding="utf-8")

    report = lightweight_existing_eval.build_report(
        summary_dir=tmp_path / "reports" / "route_summaries",
        run_roots=[tmp_path / "outputs"],
        board_paths=[],
        top_k=5,
    )

    assert "P2: 0 routes with any evidence (0 protocol-board, 0 route-summary-only)" in report
    assert "0.9100" not in report


def test_lightweight_existing_eval_selects_latest_complete_duplicate_board(monkeypatch, tmp_path):
    monkeypatch.setattr(lightweight_existing_eval, "ROOT", tmp_path)
    _write_route(tmp_path / "configs" / "routes" / "models" / "r_graph.yaml", "r_graph", "dgcnn")
    old_board = tmp_path / "outputs" / "old" / "protocol1_board.csv"
    new_board = tmp_path / "outputs" / "new" / "protocol1_board.csv"
    for board, mean in ((old_board, "0.51"), (new_board, "0.73")):
        board.parent.mkdir(parents=True)
        board.write_text(f"route_id,n_jobs,mean,std,min,max\nr_graph,25,{mean},,,\n", encoding="utf-8")
        (board.parent / "protocol1_audit.json").write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")
    old_audit = old_board.parent / "protocol1_audit.json"
    new_audit = new_board.parent / "protocol1_audit.json"
    old_time = 1_700_000_000
    new_time = 1_800_000_000
    old_audit.touch()
    new_audit.touch()
    import os

    os.utime(old_audit, (old_time, old_time))
    os.utime(new_audit, (new_time, new_time))

    report = lightweight_existing_eval.build_report(
        summary_dir=tmp_path / "reports" / "route_summaries",
        run_roots=[tmp_path / "outputs"],
        board_paths=[],
        top_k=5,
    )

    assert "0.7300" in report
    assert "0.5100" not in report


def test_lightweight_existing_eval_default_uses_committed_board_snapshots(monkeypatch, tmp_path):
    monkeypatch.setattr(lightweight_existing_eval, "ROOT", tmp_path)
    _write_route(tmp_path / "configs" / "routes" / "models" / "r_graph.yaml", "r_graph", "dgcnn")
    board = tmp_path / "reports" / "model_deep_dive" / "lightweight_existing_eval_boards" / "protocol3_board.csv"
    board.parent.mkdir(parents=True)
    board.write_text("route_id,n_jobs,mean,std,min,max\nr_graph,5,0.66,,,\n", encoding="utf-8")
    (board.parent / "protocol3_audit.json").write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")

    report = lightweight_existing_eval.build_report(
        summary_dir=tmp_path / "reports" / "route_summaries",
        run_roots=[],
        board_paths=[],
        top_k=5,
    )

    assert "P3: 1 routes with any evidence (1 protocol-board, 0 route-summary-only)" in report
    assert "0.6600" in report
