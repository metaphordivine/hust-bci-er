from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.plan_dep_hc_route_neural_jobs import FIELDS, main, plan_jobs_for_route
from scripts.run_dep_hc_router import resolve_preprocessing


def _write_route(path: Path, *, route_id: str = "fixed_crop_ea_deformer") -> Path:
    payload = {
        "route_id": route_id,
        "preprocessing": ["euclidean_alignment", "zscore"],
        "model": {
            "name": "deformer_lite",
            "conv_channels": 24,
            "transformer_depth": 2,
            "embedding_dim": 64,
            "dropout": 0.25,
        },
        "training": {"epochs": 80, "batch_size": 16},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_plan_dep_hc_route_neural_jobs_preserves_route_model_and_preprocessing(tmp_path: Path) -> None:
    route = _write_route(tmp_path / "route.yaml")
    payload = yaml.safe_load(route.read_text(encoding="utf-8"))
    payload["preprocessing"] = ["euclidean_alignment", "car", "zscore"]
    route.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    jobs = plan_jobs_for_route(
        route,
        run_prefix="screen",
        seed=42,
        p1_folds=[3],
        p2_holdout_seeds=[123, 999],
        epochs_override=8,
        batch_size_override=64,
        threshold_objective="balanced_accuracy",
        subject_aggregation="mean",
        drop_preprocessing={"euclidean_alignment"},
    )

    assert [job.run_id for job in jobs] == [
        "screen_fixed_crop_ea_deformer_p2_h123",
        "screen_fixed_crop_ea_deformer_p2_h999",
        "screen_fixed_crop_ea_deformer_p1_f3",
    ]
    assert {job.model for job in jobs} == {"deformer_lite"}
    assert {job.preprocessing for job in jobs} == {'["car","zscore"]'}
    assert {job.source_trial_sec for job in jobs} == {50.0}
    assert {job.window_sec for job in jobs} == {10.0}
    assert {job.stride_sec for job in jobs} == {None}
    assert {job.n_crops for job in jobs} == {5}
    assert {job.epochs for job in jobs} == {8}
    model_kwargs = json.loads(jobs[0].model_kwargs_json)
    assert model_kwargs["conv_channels"] == 24
    assert model_kwargs["embedding_dim"] == 64
    assert "name" not in model_kwargs


def test_plan_dep_hc_route_neural_jobs_writes_tsv(tmp_path: Path) -> None:
    route = _write_route(tmp_path / "route.yaml", route_id="fixed_crop_ea_fbcnet")
    out = tmp_path / "jobs.tsv"

    rc = main(
        [
            "--route-config",
            str(route),
            "--out",
            str(out),
            "--run-prefix",
            "dep_hc_route24",
            "--p1-fold",
            "0",
            "--p2-holdout-seed",
            "666",
            "--epochs-override",
            "4",
        ]
    )

    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == FIELDS
    rows = [dict(zip(FIELDS, line.split("\t"), strict=True)) for line in lines[1:]]
    assert len(rows) == 2
    assert rows[0]["run_id"] == "dep_hc_route24_fixed_crop_ea_fbcnet_p2_h666"
    assert rows[1]["run_id"] == "dep_hc_route24_fixed_crop_ea_fbcnet_p1_f0"
    assert rows[0]["preprocessing"] == "zscore"
    assert rows[0]["source_trial_sec"] == "50"
    assert rows[0]["window_sec"] == "10"
    assert rows[0]["stride_sec"] == "default"
    assert rows[0]["n_crops"] == "5"
    assert rows[0]["model_kwargs_json"].startswith("{")
    assert b"\r" not in out.read_bytes()


def test_plan_dep_hc_route_neural_jobs_expands_threshold_aggregation_sweep(tmp_path: Path) -> None:
    route = _write_route(tmp_path / "route.yaml", route_id="p01_wide_carz_s7")

    jobs = plan_jobs_for_route(
        route,
        run_prefix="screen",
        seed=42,
        p1_folds=[0],
        p2_holdout_seeds=[123],
        epochs_override=2,
        batch_size_override=None,
        threshold_objective=["balanced_accuracy", "min_recall"],
        subject_aggregation=["mean", "vote_frac"],
        drop_preprocessing={"euclidean_alignment"},
    )

    assert len(jobs) == 8
    assert len({job.run_id for job in jobs}) == len(jobs)
    assert "screen_p01_wide_carz_s7_thrbalanced_accuracy_aggvote_frac_p2_h123" in {
        job.run_id for job in jobs
    }
    assert {job.threshold_objective for job in jobs} == {"balanced_accuracy", "min_recall"}
    assert {job.subject_aggregation for job in jobs} == {"mean", "vote_frac"}


def test_plan_dep_hc_route_neural_jobs_can_emit_p3_inner_and_final_jobs(tmp_path: Path) -> None:
    route = _write_route(tmp_path / "route.yaml", route_id="p01_wide_carz_s7")

    jobs = plan_jobs_for_route(
        route,
        run_prefix="screen",
        seed=42,
        p1_folds=[],
        p2_holdout_seeds=[],
        p3_outer_folds=[1],
        p3_inner_folds=2,
        p3_outer_fold_count=5,
        outer_seed=42,
        inner_seed=123,
        epochs_override=2,
        batch_size_override=16,
        threshold_objective="balanced_accuracy",
        subject_aggregation="mean",
        drop_preprocessing={"euclidean_alignment"},
    )

    assert [job.run_id for job in jobs] == [
        "screen_p01_wide_carz_s7_p3_o1_i0",
        "screen_p01_wide_carz_s7_p3_o1_i1",
        "screen_p01_wide_carz_s7_p3_o1_final",
    ]
    assert {job.protocol for job in jobs} == {"p3"}
    assert [job.inner_fold for job in jobs] == [0, 1, None]
    assert {job.outer_fold for job in jobs} == {1}
    assert {job.outer_folds for job in jobs} == {5}
    assert {job.inner_folds for job in jobs} == {2}


def test_plan_dep_hc_route_neural_jobs_preserves_parameterized_preprocessing(tmp_path: Path) -> None:
    route = _write_route(tmp_path / "route.yaml")
    payload = yaml.safe_load(route.read_text(encoding="utf-8"))
    payload["preprocessing"] = [
        "euclidean_alignment",
        {"name": "bandpass", "low_hz": 1.0, "high_hz": 40.0},
        "zscore",
    ]
    route.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    jobs = plan_jobs_for_route(
        route,
        run_prefix="screen",
        seed=42,
        p1_folds=[0],
        p2_holdout_seeds=[123],
        epochs_override=None,
        batch_size_override=None,
        threshold_objective="balanced_accuracy",
        subject_aggregation="mean",
        drop_preprocessing={"euclidean_alignment"},
    )

    parsed = json.loads(jobs[0].preprocessing)
    assert parsed == [{"high_hz": 40.0, "low_hz": 1.0, "name": "bandpass"}, "zscore"]
    assert resolve_preprocessing([jobs[0].preprocessing]) == [
        {"high_hz": 40.0, "low_hz": 1.0, "name": "bandpass"},
        "zscore",
    ]


def test_plan_dep_hc_route_neural_jobs_outputs_runner_consumable_multistep_preprocessing(tmp_path: Path) -> None:
    route = _write_route(tmp_path / "route.yaml")
    payload = yaml.safe_load(route.read_text(encoding="utf-8"))
    payload["preprocessing"] = ["euclidean_alignment", "car", "zscore"]
    route.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    jobs = plan_jobs_for_route(
        route,
        run_prefix="screen",
        seed=42,
        p1_folds=[0],
        p2_holdout_seeds=[123],
        epochs_override=None,
        batch_size_override=None,
        threshold_objective="balanced_accuracy",
        subject_aggregation="vote_frac",
        drop_preprocessing={"euclidean_alignment"},
    )

    assert jobs[0].preprocessing == '["car","zscore"]'
    assert resolve_preprocessing([jobs[0].preprocessing]) == ["car", "zscore"]


def test_plan_dep_hc_route_neural_jobs_converts_sliding_window_contract(tmp_path: Path) -> None:
    route = _write_route(tmp_path / "route.yaml", route_id="sliding_ea_deformer_w6_s1")
    payload = yaml.safe_load(route.read_text(encoding="utf-8"))
    payload["input_window_sec"] = 6
    payload["augmentation"] = {
        "name": "split_first_sliding_window",
        "source_trial_sec": 10,
        "window_sec": 6,
        "stride_sec": 1,
    }
    route.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    jobs = plan_jobs_for_route(
        route,
        run_prefix="screen",
        seed=42,
        p1_folds=[0],
        p2_holdout_seeds=[123],
        epochs_override=None,
        batch_size_override=None,
        threshold_objective="balanced_accuracy",
        subject_aggregation="vote_frac",
        drop_preprocessing=None,
    )

    assert {job.source_trial_sec for job in jobs} == {10.0}
    assert {job.window_sec for job in jobs} == {6.0}
    assert {job.stride_sec for job in jobs} == {1.0}
    assert {job.n_crops for job in jobs} == {5}
    assert {job.subject_aggregation for job in jobs} == {"vote_frac"}


def test_plan_dep_hc_route_neural_jobs_uses_stable_sliding_window_count(tmp_path: Path) -> None:
    route = _write_route(tmp_path / "route.yaml", route_id="sliding_ea_deformer_w6_s0p8")
    payload = yaml.safe_load(route.read_text(encoding="utf-8"))
    payload["input_window_sec"] = 6
    payload["augmentation"] = {
        "name": "split_first_sliding_window",
        "source_trial_sec": 10,
        "window_sec": 6,
        "stride_sec": 0.8,
    }
    route.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    jobs = plan_jobs_for_route(
        route,
        run_prefix="screen",
        seed=42,
        p1_folds=[0],
        p2_holdout_seeds=[123],
        epochs_override=None,
        batch_size_override=None,
        threshold_objective="balanced_accuracy",
        subject_aggregation="vote_frac",
        drop_preprocessing=None,
    )

    assert {job.n_crops for job in jobs} == {6}


def test_plan_dep_hc_route_neural_jobs_drops_ea_by_default(tmp_path: Path) -> None:
    route = _write_route(tmp_path / "route.yaml")

    jobs = plan_jobs_for_route(
        route,
        run_prefix="screen",
        seed=42,
        p1_folds=[0],
        p2_holdout_seeds=[123],
        epochs_override=None,
        batch_size_override=None,
        threshold_objective="balanced_accuracy",
        subject_aggregation="mean",
        drop_preprocessing=None,
    )

    assert {job.preprocessing for job in jobs} == {"zscore"}


def test_plan_dep_hc_route_neural_jobs_marks_empty_preprocessing_as_none(tmp_path: Path) -> None:
    route = _write_route(tmp_path / "route.yaml")

    jobs = plan_jobs_for_route(
        route,
        run_prefix="screen",
        seed=42,
        p1_folds=[0],
        p2_holdout_seeds=[123],
        epochs_override=None,
        batch_size_override=None,
        threshold_objective="balanced_accuracy",
        subject_aggregation="mean",
        drop_preprocessing={"euclidean_alignment", "zscore"},
    )

    assert {job.preprocessing for job in jobs} == {"none"}


def test_plan_dep_hc_route_neural_jobs_rejects_unsupported_model(tmp_path: Path) -> None:
    route = _write_route(tmp_path / "route.yaml")
    payload = yaml.safe_load(route.read_text(encoding="utf-8"))
    payload["model"]["name"] = "not_a_model"
    route.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported DEP/HC neural model"):
        plan_jobs_for_route(
            route,
            run_prefix="screen",
            seed=42,
            p1_folds=[0],
            p2_holdout_seeds=[123],
            epochs_override=None,
            batch_size_override=None,
            threshold_objective="balanced_accuracy",
            subject_aggregation="mean",
            drop_preprocessing=None,
        )
