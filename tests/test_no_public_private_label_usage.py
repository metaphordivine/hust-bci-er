from pathlib import Path


SCAN_ROOTS = [Path("src"), Path("scripts"), Path("configs/routes"), Path("configs/protocols")]
BLOCKED_PATTERNS = [
    "public_label",
    "public labels",
    "private_label",
    "private labels",
    "leaderboard_feedback",
    "leaderboard feedback",
    "public_y",
    "private_y",
    "public_truth",
    "private_truth",
    "pseudo_public_gt",
]


def text_files():
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".yml", ".md", ".json", ".toml"}:
                yield path


def test_no_public_private_label_or_feedback_references_in_runtime_paths():
    hits = []
    for path in text_files():
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for pattern in BLOCKED_PATTERNS:
            if pattern in text or pattern in path.as_posix().lower():
                hits.append(f"{path}: {pattern}")
    assert hits == []
