import csv
import json
from pathlib import Path

from scripts import export_component_scores, run_score_fusion_routes


ROUTE = Path("configs/routes/models/conformer_srfnet_score_average.yaml")


def _write_score_matrix(path: Path, *, base: float) -> None:
    header = [
        "subject_id",
        "trial_id",
        "y_true",
        "crop_0",
        "crop_1",
        "crop_2",
        "crop_3",
        "crop_4",
    ]
    rows = []
    for idx in range(8):
        label = 1 if idx >= 4 else 0
        scores = [base + idx * 0.1 + crop * 0.01 for crop in range(5)]
        rows.append(["s1", f"t{idx}", str(label), *[f"{score:.3f}" for score in scores]])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_export_component_scores_prefers_score_matrix_crops(tmp_path):
    matrix = tmp_path / "score_matrix.csv"
    output = tmp_path / "component.csv"
    _write_score_matrix(matrix, base=0.0)

    rc = export_component_scores.export_component_scores_from_score_matrix(
        matrix,
        "conformer_component",
        output,
    )

    assert rc == 0
    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    assert len(rows) == 40
    assert rows[0]["component_id"] == "conformer_component"
    assert {row["crop_id"] for row in rows[:5]} == {"0", "1", "2", "3", "4"}


def test_score_fusion_runner_writes_prediction_matrix_metric_and_manifest(tmp_path):
    conformer_matrix = tmp_path / "conformer_score_matrix.csv"
    srfnet_matrix = tmp_path / "srfnet_score_matrix.csv"
    conformer_component = tmp_path / "conformer_component.csv"
    srfnet_component = tmp_path / "srfnet_component.csv"
    _write_score_matrix(conformer_matrix, base=0.0)
    _write_score_matrix(srfnet_matrix, base=1.0)
    assert export_component_scores.export_component_scores_from_score_matrix(
        conformer_matrix,
        "conformer_component",
        conformer_component,
    ) == 0
    assert export_component_scores.export_component_scores_from_score_matrix(
        srfnet_matrix,
        "srfnet_long_component",
        srfnet_component,
    ) == 0

    output_dir = tmp_path / "run"
    rc, info = run_score_fusion_routes.assemble_score_fusion(
        ROUTE,
        {
            "conformer_component": conformer_component,
            "srfnet_long_component": srfnet_component,
        },
        output_dir,
    )
    assert rc == 0
    run_score_fusion_routes._write_score_fusion_manifest(
        ROUTE,
        output_dir,
        {
            "conformer_component": conformer_component,
            "srfnet_long_component": srfnet_component,
        },
        source_manifest_paths=[],
        score_matrix_evidence=info["score_matrix_evidence"],
    )

    prediction_rows = list(csv.DictReader((output_dir / "predictions.csv").open(encoding="utf-8", newline="")))
    matrix_rows = list(csv.DictReader((output_dir / "score_matrix.csv").open(encoding="utf-8", newline="")))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert len(prediction_rows) == 8
    assert len(matrix_rows) == 8
    assert "crop_4" in matrix_rows[0]
    assert (output_dir / "metric_audit.json").exists()
    assert manifest["metric_inputs"]["score_matrix_csv"] == "score_matrix.csv"
    assert manifest["score_matrix_evidence"] == "genuine"
    assert {item["component_id"] for item in manifest["component_score_inputs"]} == {
        "conformer_component",
        "srfnet_long_component",
    }
