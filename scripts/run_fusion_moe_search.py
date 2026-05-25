from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hust_bci_er.analysis.fusion_moe_search import (  # noqa: E402
    align_score_runs,
    discover_assembled_runs,
    discover_expert_runs,
    load_score_run,
    run_posthoc_search,
    write_search_outputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run posthoc score-matrix fusion and DEP/HC soft-router search.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--assembled-glob", action="append", default=[])
    parser.add_argument("--expert-summary", type=Path)
    parser.add_argument("--expert-output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--top-experts", type=int, default=12)
    parser.add_argument("--max-base-runs", type=int, default=24)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--screen-assignments", type=int, default=4096)
    parser.add_argument("--exact-top-k", type=int, default=80)
    parser.add_argument("--fusion-asha-candidates", type=int, default=4096)
    parser.add_argument("--fusion-asha-max-width", type=int, default=4)
    parser.add_argument("--fusion-asha-eta", type=int, default=3)
    parser.add_argument("--fusion-asha-min-groups", type=int, default=48)
    parser.add_argument("--router-dep-recall", type=float, default=0.65)
    parser.add_argument("--router-hc-recall", type=float, default=0.75)
    args = parser.parse_args(argv)

    run_specs = discover_assembled_runs(args.assembled_glob, root=ROOT)
    if args.expert_summary is not None:
        run_specs.extend(
            discover_expert_runs(
                args.expert_summary,
                output_root=args.expert_output_root,
                top_n_each=args.top_experts,
            )
        )
    run_specs = [(label, paths) for label, paths in run_specs if paths]
    if len(run_specs) < 2:
        raise SystemExit("need at least two discovered runs")

    loaded = [load_score_run(label, paths) for label, paths in run_specs]
    aligned = align_score_runs(loaded)
    results = run_posthoc_search(
        aligned,
        workers=args.workers,
        max_base_runs=args.max_base_runs,
        screen_assignments=args.screen_assignments,
        exact_top_k=args.exact_top_k,
        fusion_asha_candidates=args.fusion_asha_candidates,
        fusion_asha_max_width=args.fusion_asha_max_width,
        fusion_asha_eta=args.fusion_asha_eta,
        fusion_asha_min_groups=args.fusion_asha_min_groups,
        router_dep_recall=args.router_dep_recall,
        router_hc_recall=args.router_hc_recall,
    )
    config = {
        "assembled_glob": args.assembled_glob,
        "expert_summary": str(args.expert_summary) if args.expert_summary else None,
        "expert_output_root": str(args.expert_output_root),
        "top_experts": args.top_experts,
        "max_base_runs": args.max_base_runs,
        "workers": args.workers,
        "screen_assignments": args.screen_assignments,
        "exact_top_k": args.exact_top_k,
        "fusion_asha_candidates": args.fusion_asha_candidates,
        "fusion_asha_max_width": args.fusion_asha_max_width,
        "fusion_asha_eta": args.fusion_asha_eta,
        "fusion_asha_min_groups": args.fusion_asha_min_groups,
        "router_dep_recall": args.router_dep_recall,
        "router_hc_recall": args.router_hc_recall,
        "n_loaded_runs": len(loaded),
        "n_aligned_rows": len(aligned.keys),
        "n_groups": len(aligned.groups),
        "labels": list(aligned.labels),
    }
    write_search_outputs(results, args.out_dir, config=config)
    print(json.dumps({"out_dir": str(args.out_dir), **{f"n_{k}": len(v) for k, v in results.items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
