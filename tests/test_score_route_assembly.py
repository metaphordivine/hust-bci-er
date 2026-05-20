from pathlib import Path

import numpy as np

from hust_bci_er.inference.clean_score_routes import score_route_by_id
from hust_bci_er.inference.score_fusion import assemble_score_route, assemble_score_node, calibrated_probability_average, softmax_rows
from hust_bci_er.inference.score_route_assembly import assemble_score_route_rows, write_score_route_rows


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


def test_calibrated_probability_average_returns_row_probabilities():
    first = np.array([[8, 7, 6, 5, 4, 3, 2, 1]], dtype=float)
    second = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=float)

    probs = softmax_rows(first, temperature=1.25)
    fused = calibrated_probability_average(first, second, weights=[0.75, 0.25], temperature=1.25)

    assert probs.shape == first.shape
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(fused.sum(axis=1), 1.0)
    assert fused[0, 0] > fused[0, -1]


def test_calibrated_probability_average_normalizes_explicit_weights():
    first = np.array([[8, 7, 6, 5, 4, 3, 2, 1]], dtype=float)
    second = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=float)

    percent = calibrated_probability_average(first, second, weights=[75, 25], temperature=1.25)
    normalized = calibrated_probability_average(first, second, weights=[0.75, 0.25], temperature=1.25)

    assert np.allclose(percent, normalized)
    assert np.allclose(percent.sum(axis=1), 1.0)


def test_calibrated_probability_average_rejects_invalid_weights():
    first = np.array([[8, 7, 6, 5, 4, 3, 2, 1]], dtype=float)
    second = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=float)

    for weights, message in [
        ([0.0, 0.0], "weights sum must be positive"),
        ([1.0, -0.1], "weights must be non-negative"),
        ([1.0, np.nan], "weights must be finite"),
    ]:
        try:
            calibrated_probability_average(first, second, weights=weights)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"invalid weights should fail: {weights}")


def test_interpretable_calibrated_diverse_route_from_component_arrays():
    route = score_route_by_id("interpretable_calibrated_diverse_score_fusion")
    assert route is not None
    component_scores = {
        "fixed_crop_ea_fbstcnet_component": np.array([[8, 7, 6, 5, 4, 3, 2, 1]], dtype=float),
        "srfnet_long_component": np.array([[7, 8, 6, 5, 4, 3, 2, 1]], dtype=float),
        "conformer_component": np.array([[8, 6, 7, 5, 4, 3, 2, 1]], dtype=float),
        "dgcnn_dann_cohort_component": np.array([[6, 7, 8, 5, 4, 3, 2, 1]], dtype=float),
        "dgcnn_coral_cohort_component": np.array([[7, 6, 8, 5, 4, 3, 2, 1]], dtype=float),
        "riemannian_tangent_component": np.array([[5, 6, 7, 8, 4, 3, 2, 1]], dtype=float),
    }

    scores = assemble_score_route(route, component_scores)

    assert scores.shape == (1, 8)
    assert np.isfinite(scores).all()
    assert np.allclose(scores.sum(axis=1), 1.0)


def test_assemble_score_route_rows_from_component_csvs(tmp_path):
    route_config = Path("configs/routes/models/conformer_srfnet_score_average.yaml")
    conformer = tmp_path / "conformer.csv"
    srfnet = tmp_path / "srfnet.csv"
    header = "component_id,subject_id,trial_id,score\n"
    conformer.write_text(header + "\n".join(f"conformer_component,s1,t{idx},{8 - idx}" for idx in range(8)) + "\n", encoding="utf-8")
    srfnet.write_text(header + "\n".join(f"srfnet_long_component,s1,t{idx},{7 - idx}" for idx in range(8)) + "\n", encoding="utf-8")

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
    header = "component_id,seed,fold,subject_id,trial_id,score\n"
    conformer_lines = []
    srfnet_lines = []
    for seed in (1, 2):
        for idx in range(8):
            conformer_lines.append(f"conformer_component,{seed},0,s1,t{idx},{8 - idx + seed * 0.01}")
            srfnet_lines.append(f"srfnet_long_component,{seed},0,s1,t{idx},{7 - idx + seed * 0.01}")
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


def test_assemble_score_route_rows_uses_crop_id_alignment_key(tmp_path):
    route_config = Path("configs/routes/models/conformer_srfnet_score_average.yaml")
    conformer = tmp_path / "conformer.csv"
    srfnet = tmp_path / "srfnet.csv"
    header = "component_id,subject_id,trial_id,crop_id,score\n"
    conformer_lines = []
    srfnet_lines = []
    for crop_id in (0, 1):
        for idx in range(8):
            conformer_lines.append(f"conformer_component,s1,t{idx},{crop_id},{8 - idx + crop_id * 0.01}")
            srfnet_lines.append(f"srfnet_long_component,s1,t{idx},{crop_id},{7 - idx + crop_id * 0.01}")
    conformer.write_text(header + "\n".join(conformer_lines) + "\n", encoding="utf-8")
    srfnet.write_text(header + "\n".join(srfnet_lines) + "\n", encoding="utf-8")

    rows = assemble_score_route_rows(
        route_config,
        {
            "conformer_component": conformer,
            "srfnet_long_component": srfnet,
        },
    )
    out = tmp_path / "predictions.csv"
    write_score_route_rows(rows, out)

    assert len(rows) == 16
    assert {row["crop_id"] for row in rows} == {"0", "1"}
    assert sum(int(row["pred_top4"]) for row in rows if row["crop_id"] == "0") == 4
    assert sum(int(row["pred_top4"]) for row in rows if row["crop_id"] == "1") == 4
    assert "crop_id" in out.read_text(encoding="utf-8").splitlines()[0]


def test_assemble_score_route_rows_preserves_consistent_y_true_and_component_id(tmp_path):
    route_config = Path("configs/routes/models/conformer_srfnet_score_average.yaml")
    conformer = tmp_path / "conformer.csv"
    srfnet = tmp_path / "srfnet.csv"
    header = "component_id,subject_id,trial_id,score,y_true\n"
    conformer.write_text(
        header + "\n".join(f"conformer_component,s1,t{idx},{8 - idx},{1 if idx < 4 else 0}" for idx in range(8)) + "\n",
        encoding="utf-8",
    )
    srfnet.write_text(
        header + "\n".join(f"srfnet_long_component,s1,t{idx},{7 - idx},{1 if idx < 4 else 0}" for idx in range(8)) + "\n",
        encoding="utf-8",
    )

    rows = assemble_score_route_rows(
        route_config,
        {
            "conformer_component": conformer,
            "srfnet_long_component": srfnet,
        },
    )
    out = tmp_path / "predictions.csv"
    write_score_route_rows(rows, out)

    assert {row["y_true"] for row in rows[:4]} == {"1"}
    assert {row["y_true"] for row in rows[4:]} == {"0"}
    assert "y_true" in out.read_text(encoding="utf-8").splitlines()[0]


def test_assemble_score_route_rows_rejects_component_id_mismatch(tmp_path):
    route_config = Path("configs/routes/models/conformer_srfnet_score_average.yaml")
    conformer = tmp_path / "conformer.csv"
    srfnet = tmp_path / "srfnet.csv"
    header = "component_id,subject_id,trial_id,score\n"
    conformer.write_text(header + "\n".join(f"srfnet_long_component,s1,t{idx},{8 - idx}" for idx in range(8)) + "\n", encoding="utf-8")
    srfnet.write_text(header + "\n".join(f"srfnet_long_component,s1,t{idx},{7 - idx}" for idx in range(8)) + "\n", encoding="utf-8")

    try:
        assemble_score_route_rows(
            route_config,
            {
                "conformer_component": conformer,
                "srfnet_long_component": srfnet,
            },
        )
    except ValueError as exc:
        assert "component_id does not match conformer_component" in str(exc)
    else:
        raise AssertionError("component_id mismatch should fail")


def test_assemble_score_route_rows_rejects_missing_or_empty_component_id(tmp_path):
    route_config = Path("configs/routes/models/conformer_srfnet_score_average.yaml")
    conformer = tmp_path / "conformer.csv"
    srfnet = tmp_path / "srfnet.csv"
    conformer.write_text("subject_id,trial_id,score\n" + "\n".join(f"s1,t{idx},{8 - idx}" for idx in range(8)) + "\n", encoding="utf-8")
    srfnet.write_text("component_id,subject_id,trial_id,score\n" + "\n".join(f"srfnet_long_component,s1,t{idx},{7 - idx}" for idx in range(8)) + "\n", encoding="utf-8")

    try:
        assemble_score_route_rows(
            route_config,
            {
                "conformer_component": conformer,
                "srfnet_long_component": srfnet,
            },
        )
    except ValueError as exc:
        assert "must include component_id" in str(exc)
    else:
        raise AssertionError("missing component_id column should fail")

    conformer.write_text("component_id,subject_id,trial_id,score\n" + "\n".join(f"{'' if idx == 0 else 'conformer_component'},s1,t{idx},{8 - idx}" for idx in range(8)) + "\n", encoding="utf-8")
    try:
        assemble_score_route_rows(
            route_config,
            {
                "conformer_component": conformer,
                "srfnet_long_component": srfnet,
            },
        )
    except ValueError as exc:
        assert "component_id is empty" in str(exc)
    else:
        raise AssertionError("empty component_id should fail")


def test_assemble_score_route_rows_rejects_y_true_mismatch(tmp_path):
    route_config = Path("configs/routes/models/conformer_srfnet_score_average.yaml")
    conformer = tmp_path / "conformer.csv"
    srfnet = tmp_path / "srfnet.csv"
    header = "component_id,subject_id,trial_id,score,y_true\n"
    conformer.write_text(
        header + "\n".join(f"conformer_component,s1,t{idx},{8 - idx},{1 if idx < 4 else 0}" for idx in range(8)) + "\n",
        encoding="utf-8",
    )
    srfnet.write_text(
        header + "\n".join(f"srfnet_long_component,s1,t{idx},{7 - idx},{0 if idx == 0 else (1 if idx < 4 else 0)}" for idx in range(8)) + "\n",
        encoding="utf-8",
    )

    try:
        assemble_score_route_rows(
            route_config,
            {
                "conformer_component": conformer,
                "srfnet_long_component": srfnet,
            },
        )
    except ValueError as exc:
        assert "y_true mismatch" in str(exc)
    else:
        raise AssertionError("y_true mismatch should fail")


def test_assemble_score_route_rows_rejects_partial_y_true_when_any_component_provides_truth(tmp_path):
    route_config = Path("configs/routes/models/conformer_srfnet_score_average.yaml")
    conformer = tmp_path / "conformer.csv"
    srfnet = tmp_path / "srfnet.csv"
    conformer.write_text(
        "component_id,subject_id,trial_id,score,y_true\n"
        + "\n".join(f"conformer_component,s1,t{idx},{8 - idx},{1 if idx < 4 else 0}" for idx in range(8))
        + "\n",
        encoding="utf-8",
    )
    srfnet.write_text(
        "component_id,subject_id,trial_id,score\n" + "\n".join(f"srfnet_long_component,s1,t{idx},{7 - idx}" for idx in range(8)) + "\n",
        encoding="utf-8",
    )

    try:
        assemble_score_route_rows(
            route_config,
            {
                "conformer_component": conformer,
                "srfnet_long_component": srfnet,
            },
        )
    except ValueError as exc:
        assert "missing y_true" in str(exc)
    else:
        raise AssertionError("missing y_true in one component should fail when any component provides truth")
