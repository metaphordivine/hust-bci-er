from pathlib import Path

from scripts.update_route_board import generate_board


def test_route_board_is_generated_from_route_configs():
    assert Path("reports/route_board.md").read_text(encoding="utf-8") == generate_board()
