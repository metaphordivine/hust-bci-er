from pathlib import Path

from scripts import check_route_registry
from scripts.update_route_board import generate_board


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
