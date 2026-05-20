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
    assert "| IDEA |  |" in row
    assert "| pending_placeholder |" in row
