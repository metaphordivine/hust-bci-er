from scripts.audit_experiment import MISSING_MODEL_IMPL_MAPPING_PREFIX, route_implementation_paths


def test_new_models_have_candidate_audit_implementation_paths():
    expected = {
        "tsception": "src/hust_bci_er/models/backbones/tsception.py",
        "fbcnet": "src/hust_bci_er/models/backbones/fbcnet.py",
        "dgcnn": "src/hust_bci_er/models/graph/dgcnn.py",
        "lggnet": "src/hust_bci_er/models/graph/lggnet.py",
    }
    for model_name, model_path in expected.items():
        paths = route_implementation_paths(
            {
                "model": {"name": model_name},
                "training": {"job_adapter": "torch_classifier"},
            },
            f"configs/routes/models/fixed_crop_ea_{model_name}.yaml",
        )
        assert model_path in paths
        assert not any(path.startswith(MISSING_MODEL_IMPL_MAPPING_PREFIX) for path in paths)
