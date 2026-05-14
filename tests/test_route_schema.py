from pathlib import Path

from scripts.validate_route import validate_file


def test_example_route_schema_passes():
    path = Path("configs/routes/models/ea_deformer.yaml")
    assert validate_file(path) == []
