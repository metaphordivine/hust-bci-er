from pathlib import Path

import numpy as np

from hust_bci_er.inference.clean_score_routes import score_route_by_id
from hust_bci_er.inference.score_fusion import assemble_score_route, assemble_score_node
from hust_bci_er.inference.score_route_assembly import assemble_score_route_rows


def test_assemble_score_route_from_component_arrays():
    route = score_route_by_id("conformer_srfnet_score_average")
    assert route is not None
    component_scores = {
        "conformer_component": np.array([[8, 7, 6, 5, 4, 3, 2, 1]], dtype=float),
        "srfnet_long_component": np.array([[7, 8, 6, 5, 4, 3, 2, 1]], dtype=float),
    }

    scores = assemble_score_route(route, component_scores)
    assert scores.shape == (1, 8)
    assert np.isfinite(scores).all()
    assert np.array_equal(scores, assemble_score_node(route.score_node, component_scores))


def test_assemble_score_route_rows_from_component_csvs(tmp_path):
    route_config = Path("configs/routes/models/conformer_srfnet_score_average.yaml")
    conformer = tmp_path / "conformer.csv"
    srfnet = tmp_path / "srfnet.csv"
    header = "subject_id,trial_id,score\n"
    conformer.write_text(header + "\n".join(f"s1,t{idx},{8 - idx}" for idx in range(8)) + "\n", encoding="utf-8")
    srfnet.write_text(header + "\n".join(f"s1,t{idx},{7 - idx}" for idx in range(8)) + "\n", encoding="utf-8")

    rows = assemble_score_route_rows(
        route_config,
        {
            "conformer_component": conformer,
            "srfnet_long_component": srfnet,
        },
    )

    assert len(rows) == 8
    assert {row["route_id"] for row in rows} == {"conformer_srfnet_score_average"}
    assert sum(int(row["pred_top4"]) for row in rows) == 4


def test_assemble_score_route_rows_uses_seed_fold_alignment_keys(tmp_path):
    route_config = Path("configs/routes/models/conformer_srfnet_score_average.yaml")
    conformer = tmp_path / "conformer.csv"
    srfnet = tmp_path / "srfnet.csv"
    header = "seed,fold,subject_id,trial_id,score\n"
    conformer_lines = []
    srfnet_lines = []
    for seed in (1, 2):
        for idx in range(8):
            conformer_lines.append(f"{seed},0,s1,t{idx},{8 - idx + seed * 0.01}")
            srfnet_lines.append(f"{seed},0,s1,t{idx},{7 - idx + seed * 0.01}")
    conformer.write_text(header + "\n".join(conformer_lines) + "\n", encoding="utf-8")
    srfnet.write_text(header + "\n".join(srfnet_lines) + "\n", encoding="utf-8")

    rows = assemble_score_route_rows(
        route_config,
        {
            "conformer_component": conformer,
            "srfnet_long_component": srfnet,
        },
    )

    assert len(rows) == 16
    assert {row["seed"] for row in rows} == {"1", "2"}
    assert {row["fold"] for row in rows} == {"0"}
    assert sum(int(row["pred_top4"]) for row in rows if row["seed"] == "1") == 4
    assert sum(int(row["pred_top4"]) for row in rows if row["seed"] == "2") == 4
