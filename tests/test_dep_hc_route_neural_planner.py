from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.plan_dep_hc_route_neural_jobs import FIELDS, main, plan_jobs_for_route


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
    assert {job.preprocessing for job in jobs} == {"zscore"}
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
    assert rows[0]["model_kwargs_json"].startswith("{")
    assert b"\r" not in out.read_bytes()


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
