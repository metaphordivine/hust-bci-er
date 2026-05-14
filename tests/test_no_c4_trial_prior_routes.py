from pathlib import Path


def test_no_c4_trial_prior_routes_in_configs_or_registry():
    checked_roots = [Path("configs/routes"), Path("src/hust_bci_er/inference")]
    banned = "c4_prior_component"
    offenders = []
    for root in checked_roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}:
                if banned in path.read_text(encoding="utf-8"):
                    offenders.append(str(path))
    assert offenders == []

