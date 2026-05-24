from pathlib import Path

from scripts import check_route_registry
from scripts.update_route_board import (
    format_score_items,
    generate_board,
    markdown_bullet_score_items,
)


def test_route_board_is_generated_from_route_configs():
    assert Path("reports/route_board.md").read_text(encoding="utf-8") == generate_board()


def test_route_registry_is_synchronized():
    assert check_route_registry.main() == 0


def test_pending_placeholder_summary_is_not_counted_as_latest_gate():
    board = generate_board()
    row = next(line for line in board.splitlines() if line.startswith("| `tuned_sliding_window_fbstcnet` |"))
    assert "| PENDING | IDEA | IDEA | PENDING_REMOTE_P1 |  |" in row
    assert "| pending_placeholder |" in row


def test_remote_protocol_scores_update_board_state_without_promoting_route():
    board = generate_board()
    row = next(line for line in board.splitlines() if line.startswith("| `fixed_crop_ea_whitening_eps3e4_fbstcnet` |"))
    assert "| REMOTE_SCORES_PRESENT | IDEA | IDEA | PENDING_REMOTE_PROTOCOL_RUNS |  |" in row
    assert "| pending_placeholder | summary_refresh_pending |" in row
    assert "remote_protocol_score_ledger_20260523:B:P2" in board


def test_route_score_ledger_includes_committed_remote_protocol_scores():
    board = generate_board()
    ledger = board.split("## Route Score Ledger", 1)[1]
    assert "| `fixed_crop_whitening_eps3e4_fbstcnet_m_conn_light` | REMOTE_SCORES_PRESENT |" in ledger
    assert "remote_protocol_score_ledger_20260523" in ledger


def test_score_bullet_extraction_skips_governance_text():
    text = "\n".join(
        [
            "- Candidate audit: PASS with genuine component score evidence and 20 component source manifests.",
            "- Promote only if P1/P2 evidence improves without worsening DEP/HC balance.",
            "- P1 seed42 exact BA: 0.7161.",
            "- DEP mean BA: 0.6119; HC mean BA: 0.7683; DEP/HC ratio: 0.7965.",
        ]
    )

    items = markdown_bullet_score_items(text)

    assert "P1 seed42 exact BA: 0.7161" in items
    assert "DEP mean BA: 0.6119; HC mean BA: 0.7683; DEP/HC ratio: 0.7965" in items
    assert not any("Candidate audit" in item for item in items)
    assert not any("Promote only" in item for item in items)


def test_score_ledger_code_spans_tolerate_embedded_backticks():
    formatted = format_score_items(["P3 final mean BA: 0.6294 from `outputs/run`"])

    assert formatted == "`` P3 final mean BA: 0.6294 from `outputs/run` ``"
