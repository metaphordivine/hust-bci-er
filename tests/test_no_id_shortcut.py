import yaml


def test_example_route_does_not_use_id_features():
    with open("configs/routes/models/ea_deformer.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    features = set(data.get("features") or [])
    id_features = {"subject_id", "trial_id", "pseudo_trial_id", "user_id"}
    assert features.isdisjoint(id_features)
