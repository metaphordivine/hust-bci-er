from pathlib import Path

import yaml


def test_protocols_are_subject_grouped():
    for path in Path("configs/protocols").glob("*.yaml"):
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("subject_group_split") is True

