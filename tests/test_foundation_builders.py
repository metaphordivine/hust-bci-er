import csv
from pathlib import Path

import yaml

from hust_bci_er.audit.manifest import sha256_file
from hust_bci_er.data.loader_contract import crops_from_manifest
from hust_bci_er.data.manifest_builder import build_dataset_manifest, read_index_csv, write_dataset_manifest
from hust_bci_er.data.qa import dataset_qa, render_qa_markdown
from hust_bci_er.data.split_builder import build_split_manifest, write_split_manifest


def write_index(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "subject_id", "trial_id", "crop_id", "y_true"])
        writer.writeheader()
        writer.writerows(rows)


def test_dataset_manifest_builder_indexes_sources_and_checksums(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "s1_t1_c0.npy").write_bytes(b"a")
    (data_dir / "s2_t1_c0.npy").write_bytes(b"b")
    index = tmp_path / "index.csv"
    write_index(
        index,
        [
            {"path": "data/s1_t1_c0.npy", "subject_id": "s1", "trial_id": "t1", "crop_id": 0, "y_true": 1},
            {"path": "data/s2_t1_c0.npy", "subject_id": "s2", "trial_id": "t1", "crop_id": 0, "y_true": 0},
        ],
    )

    manifest = build_dataset_manifest(dataset_version="toy_v1", rows=read_index_csv(index), base_dir=tmp_path)

    assert manifest["status"] == "ready"
    assert manifest["subject_ids"] == ["s1", "s2"]
    assert manifest["n_trials"] == 2
    assert manifest["n_crops"] == 1
    assert manifest["checksum_manifest"][0]["sha256"] == sha256_file(data_dir / "s1_t1_c0.npy")
    assert manifest["trial_index"][0]["label_available"] is True
    crops = crops_from_manifest(manifest)
    assert crops[0].subject_id == "s1"
    assert crops[0].crop_id == 0

    output = tmp_path / "dataset.yaml"
    write_dataset_manifest(manifest, output)
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["dataset_version"] == "toy_v1"


def test_split_manifest_builder_is_seeded_and_subject_level(tmp_path):
    manifest = {
        "dataset_version": "toy_v1",
        "trial_index": [
            {"subject_id": "s1", "trial_id": "t1"},
            {"subject_id": "s2", "trial_id": "t1"},
            {"subject_id": "s3", "trial_id": "t1"},
            {"subject_id": "s4", "trial_id": "t1"},
        ],
    }

    split = build_split_manifest(
        split_id="toy_split",
        dataset=manifest,
        seed=7,
        val_count=1,
        test_count=1,
        dataset_manifest_path="configs/datasets/toy.yaml",
        dataset_manifest_sha256="1" * 64,
    )
    split_again = build_split_manifest(
        split_id="toy_split",
        dataset=manifest,
        seed=7,
        val_count=1,
        test_count=1,
        dataset_manifest_path="configs/datasets/toy.yaml",
        dataset_manifest_sha256="1" * 64,
    )

    assert split == split_again
    assert split["dataset_version"] == "toy_v1"
    assert split["dataset_manifest_sha256"] == "1" * 64
    assert set(split["train_subjects"]).isdisjoint(split["val_subjects"])
    assert set(split["train_subjects"]).isdisjoint(split["test_subjects"])
    assert {row["split"] for row in split["trial_rows"]} == {"train", "val", "test"}

    output = tmp_path / "split.yaml"
    write_split_manifest(split, output)
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["split_id"] == "toy_split"


def test_dataset_qa_reports_counts_and_checksum_gaps():
    manifest = {
        "dataset_version": "toy_v1",
        "data_sources": [{"path": "a"}, {"path": "b"}],
        "checksum_manifest": [{"path": "a", "sha256": "1" * 64}, {"path": "old", "sha256": "2" * 64}],
        "trial_index": [
            {"subject_id": "s1", "trial_id": "t1", "crop_id": 0, "y_true": 1},
            {"subject_id": "s1", "trial_id": "t1", "crop_id": 1, "y_true": 1},
            {"subject_id": "s2", "trial_id": "t2", "crop_id": 0, "y_true": 0},
        ],
    }

    report = dataset_qa(manifest)

    assert report["n_subjects"] == 2
    assert report["n_trials"] == 2
    assert report["n_crops"] == 3
    assert report["n_unique_crop_ids"] == 2
    assert report["missing_checksum_paths"] == ["b"]
    assert report["extra_checksum_paths"] == ["old"]
    assert "Dataset QA Report" in render_qa_markdown(report)


def test_dataset_manifest_builder_marks_missing_sources_without_fake_checksum(tmp_path):
    manifest = build_dataset_manifest(
        dataset_version="toy_v1",
        rows=[{"path": "missing.npy", "subject_id": "s1", "trial_id": "t1", "crop_id": 0}],
        base_dir=tmp_path,
        require_files=False,
    )

    assert manifest["status"] == "declared_with_missing_data_sources"
    assert manifest["missing_data_sources"] == ["missing.npy"]
    assert manifest["checksum_manifest"] == []
    assert manifest["data_sources"][0]["checksum_available"] is False
