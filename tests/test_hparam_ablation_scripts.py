from __future__ import annotations

import csv
from pathlib import Path

import yaml

import scripts.hparam_search as hparam_search
import scripts.run_ablation as run_ablation


def test_hparam_select_best_reads_flat_csv_metrics() -> None:
    rows = [
        {
            "training.optimizer.lr": "0.001",
            "error": "False",
            "metric_exact_single_crop_expected_BA": "0.61",
        },
        {
            "training.optimizer.lr": "0.01",
            "error": "False",
            "metric_exact_single_crop_expected_BA": "0.73",
        },
        {
            "training.optimizer.lr": "0.0001",
            "error": "True",
            "metric_exact_single_crop_expected_BA": "0.99",
        },
    ]

    best = hparam_search._select_best(rows)

    assert best["training.optimizer.lr"] == "0.01"


def test_hparam_select_best_reads_live_primary_metric_dict() -> None:
    rows = [
        {
            "combo": {"training.optimizer.lr": 0.001},
            "error": False,
            "primary_metric": {"exact_single_crop_expected_BA": 0.61},
        },
        {
            "combo": {"training.optimizer.lr": 0.01},
            "error": False,
            "primary_metric": {"exact_single_crop_expected_BA": 0.73},
        },
    ]

    best = hparam_search._select_best(rows)

    assert best["combo"]["training.optimizer.lr"] == 0.01


def test_hparam_results_csv_handles_first_row_error_then_metric(tmp_path: Path) -> None:
    csv_path = tmp_path / "coarse_results.csv"
    hparam_search._results_to_csv(
        [
            {
                "combo": {"training.optimizer.lr": 0.001},
                "error": True,
                "stderr": "route validation failed",
            },
            {
                "combo": {"training.optimizer.lr": 0.01},
                "primary_metric": {"exact_single_crop_expected_BA": 0.73},
                "run_dir": "run_001",
            },
        ],
        csv_path,
        ["training.optimizer.lr"],
    )

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))

    assert rows[0]["error"] == "True"
    assert rows[1]["error"] == "False"
    assert rows[1]["metric_value"] == "0.73"
    assert rows[1]["metric_exact_single_crop_expected_BA"] == "0.73"


def test_hparam_fine_grid_uses_neighborhood_around_best() -> None:
    combos = hparam_search._generate_fine_combinations(
        {
            "parameters": {
                "training.optimizer.lr": {
                    "coarse": [0.0001, 0.001, 0.01],
                    "fine": [0.0005, 0.00075, 0.001, 0.0025, 0.005],
                }
            }
        },
        {"training.optimizer.lr": "0.01"},
    )

    values = [combo["training.optimizer.lr"] for combo in combos]
    assert values == [0.0025, 0.005]
    assert 0.0005 not in values


def test_ablation_baseline_route_id_matches_temp_file(tmp_path: Path) -> None:
    route_path = run_ablation._build_temp_route_config(
        {"route_id": "sliding_window_eegnet", "training": {"epochs": 1}},
        tmp_path,
        "baseline_route",
    )

    data = yaml.safe_load(route_path.read_text(encoding="utf-8"))

    assert route_path.name == "baseline_route.yaml"
    assert data["route_id"] == "baseline_route"


def test_ablation_stops_when_baseline_fails(monkeypatch, tmp_path: Path) -> None:
    route = tmp_path / "route.yaml"
    route.write_text("route_id: sliding_window_eegnet\n", encoding="utf-8")
    ablations = tmp_path / "ablations.yaml"
    ablations.write_text(
        """
variants:
  - name: low_lr
    overrides:
      training.optimizer.lr: 0.0005
""".lstrip(),
        encoding="utf-8",
    )

    calls: list[Path] = []

    def fake_run_single(**kwargs):
        calls.append(kwargs["temp_route"])
        return {"error": True, "stderr": "baseline failed"}

    monkeypatch.setattr(run_ablation, "_run_single", fake_run_single)

    exit_code = run_ablation.main([
        "--route",
        str(route),
        "--ablations",
        str(ablations),
        "--run-dir",
        str(tmp_path / "out"),
        "--data-root",
        "data",
    ])

    assert exit_code == 1
    assert len(calls) == 1
    assert calls[0].name == "baseline_route.yaml"


def test_ablation_invalid_metric_is_error_in_results_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "ablation_results.csv"

    run_ablation._write_results_csv(
        [
            {
                "variant": "bad_metric",
                "description": "invalid metric",
                "primary_metric": "not-json-or-float",
            }
        ],
        csv_path,
        baseline_metric=0.5,
    )

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    assert rows[0]["error"] == "True"
    assert rows[0]["metric_value"] == ""
    assert rows[0]["delta_vs_baseline"] == ""
    assert "primary_metric" in rows[0]["error_detail"]


def test_ablation_extract_metric_rejects_empty_dict() -> None:
    try:
        run_ablation._extract_metric_value({})
    except ValueError:
        pass
    else:
        raise AssertionError("empty metric dict must not be coerced to 0.0")
