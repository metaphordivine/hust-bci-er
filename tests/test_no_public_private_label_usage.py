from pathlib import Path


def test_no_public_private_label_artifacts_committed():
    banned_names = {"public_labels.csv", "private_labels.csv", "leaderboard_feedback.csv"}
    paths = [p.name.lower() for p in Path(".").rglob("*") if ".git" not in p.parts]
    assert not banned_names.intersection(paths)

