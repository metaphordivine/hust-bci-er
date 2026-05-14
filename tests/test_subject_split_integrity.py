from pathlib import Path

import yaml

from hust_bci_er.data.splits import assert_disjoint_subjects, assert_original_trial_not_cross_split


def test_protocols_are_subject_grouped():
    for path in Path("configs/protocols").glob("*.yaml"):
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("subject_group_split") is True


def test_declared_split_manifests_have_no_subject_or_trial_overlap():
    split_root = Path("configs/splits")
    assert split_root.exists()
    for path in split_root.glob("*.yaml"):
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        assert data.get("split_id") == path.stem
        assert data.get("subject_group_split") is True
        assert_disjoint_subjects(
            set(map(str, data.get("train_subjects") or [])),
            set(map(str, data.get("val_subjects") or [])),
            set(map(str, data.get("test_subjects") or [])),
        )
        rows = data.get("trial_rows") or []
        if rows:
            assert_original_trial_not_cross_split(rows)


def test_route_split_ids_have_manifests():
    split_ids = {path.stem for path in Path("configs/splits").glob("*.yaml")}
    for path in Path("configs/routes/models").glob("*.yaml"):
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        assert data.get("split_id") in split_ids


def test_route_dataset_versions_have_manifests():
    dataset_root = Path("configs/datasets")
    assert dataset_root.exists()
    dataset_versions = {path.stem for path in dataset_root.glob("*.yaml")}
    for path in Path("configs/routes/models").glob("*.yaml"):
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        assert data.get("dataset_version") in dataset_versions
