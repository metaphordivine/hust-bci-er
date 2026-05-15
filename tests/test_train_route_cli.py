from __future__ import annotations

from types import SimpleNamespace

import scripts.train_route as train_route


def test_train_route_passes_data_root_to_adapter(monkeypatch, tmp_path):
    route = tmp_path / "route.yaml"
    route.write_text(
        "\n".join(
            [
                "route_id: r1",
                "training:",
                "  job_adapter: torch_classifier",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    data_root = tmp_path / "hust_data"
    captured = {}

    def fake_adapter(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            run_dir=kwargs["run_dir"],
            manifest_json=kwargs["run_dir"] / "manifest.json",
            metric_report={"metrics": {"dummy": 1.0}},
        )

    monkeypatch.setattr(train_route, "_resolve_adapter", lambda name: fake_adapter)

    exit_code = train_route.main(
        [
            "--mode",
            "candidate",
            "--route",
            str(route),
            "--run-dir",
            str(run_dir),
            "--data-root",
            str(data_root),
        ]
    )

    assert exit_code == 0
    assert captured["data_root"] == data_root
    assert "--data-root" in captured["command"]
    assert data_root.as_posix() in captured["command"]
