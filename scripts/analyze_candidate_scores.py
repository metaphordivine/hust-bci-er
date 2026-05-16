from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.analysis.candidate_scores import analyze_score_matrices, write_markdown_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze candidate score matrices for leaderboard, crop, calibration, and fusion diagnostics.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="Run directories, parent directories, or score_matrix.csv files.")
    parser.add_argument("--out", type=Path, required=True, help="Markdown report path.")
    parser.add_argument("--top-errors", type=int, default=40, help="Maximum frequent error samples to include.")
    args = parser.parse_args(argv)

    analysis = analyze_score_matrices(args.runs, top_errors=args.top_errors)
    write_markdown_report(analysis, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
