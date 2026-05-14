from pathlib import Path


def test_route_names_do_not_use_historical_shorthand():
    banned_terms = {
        "s" + "9",
        "z" + "av" + "g",
        "conf_" + "z" + "av" + "g",
        "heur" + "istic",
        "bl" + "end",
        "qkv_" + "best",
        "old_" + "whitening",
        "h0" + "25",
        "h0" + "30",
        "h0" + "40",
        "high_" + "value",
    }
    offenders = []
    roots = [Path("configs/routes"), Path("src/hust_bci_er/inference"), Path("reports/route_board.md")]
    for root in roots:
        paths = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for path in paths:
            if path.suffix not in {".py", ".yaml", ".yml", ".md"}:
                continue
            text = path.read_text(encoding="utf-8").lower()
            for term in banned_terms:
                if term in path.name.lower() or term in text:
                    offenders.append(f"{path}:{term}")
    assert offenders == []
