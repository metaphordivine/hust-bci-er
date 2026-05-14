"""Central registry for valid route component names."""

ROUTE_STATUSES = {
    "IDEA",
    "SMOKE_ONLY",
    "DIAGNOSTIC_ONLY",
    "CANDIDATE",
    "PROMOTED",
    "REJECTED",
    "ARCHIVED",
}

PREPROCESSING = {
    "zscore",
    "euclidean_alignment",
    "whitening_eps1e3",
    "whitening_eps3e4",
    "robust_zscore",
    "shrinkage_whitening",
    "car",
    "bandpass",
}

FEATURES = {
    "bandpower",
    "differential_entropy",
    "hjorth",
    "connectivity",
}

MODELS = {
    "eegnet",
    "conformer_lite",
    "deformer_lite",
    "srfnet",
    "random_forest",
    "extra_trees",
    "logistic_regression",
    "lggnet",
    "score_fusion",
}

ADAPTATION = {
    "none",
    "adabn",
    "dann",
}

EVALUATION_PROTOCOLS = {
    "p1_repeated_group_kfold",
    "p2_pseudo_public_holdout",
    "p3_nested_selection",
}

PRIMARY_METRICS = {
    "exact_single_crop_expected_BA",
    "top4_BA",
    "no_top4_BA",
    "all_correct_rate",
}
