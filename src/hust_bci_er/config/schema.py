"""Small route schema validator.

This intentionally avoids a heavy validation framework for the initial skeleton.
It can be replaced by Pydantic later without changing the command contract.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

from . import registry
from hust_bci_er.inference.clean_score_routes import score_route_by_id


REQUIRED_TOP_LEVEL = {
    "route_id",
    "status",
    "dataset_version",
    "split_id",
    "seed",
    "input_window_sec",
    "model",
    "inference",
    "evaluation",
}
DEFAULT_SFREQ = 250.0


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def model_name(model: Any) -> str | None:
    if isinstance(model, str):
        return model
    if isinstance(model, dict):
        name = model.get("name")
        return str(name) if name is not None else None
    return None


def nested_name(value: Any, key: str = "name") -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        item = value.get(key)
        return str(item) if item is not None else None
    return None


def preprocessing_component_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        name = value.get("name")
        return str(name) if name is not None else None
    return None


def validate_route_config(data: dict[str, Any], path: Path | None = None) -> list[str]:
    errors: list[str] = []

    missing = sorted(REQUIRED_TOP_LEVEL.difference(data))
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")

    route_id = data.get("route_id")
    if not isinstance(route_id, str) or not route_id:
        errors.append("route_id must be a non-empty string")
    elif path is not None and path.suffix in {".yaml", ".yml"} and path.stem != route_id:
        errors.append(f"route_id must match file name: {path.stem}")

    status = data.get("status")
    if status not in registry.ROUTE_STATUSES:
        errors.append(f"invalid status: {status}")

    seed = data.get("seed")
    if not isinstance(seed, int):
        errors.append("seed must be an integer")

    input_window_sec = data.get("input_window_sec")
    if not isinstance(input_window_sec, (int, float)) or input_window_sec <= 0:
        errors.append("input_window_sec must be positive")

    for item in as_list(data.get("preprocessing")):
        name = preprocessing_component_name(item)
        if name not in registry.PREPROCESSING:
            errors.append(f"unknown preprocessing component: {name}")
        elif isinstance(item, dict):
            validate_preprocessing_spec(item, errors)

    for name in as_list(data.get("features")):
        if name not in registry.FEATURES:
            errors.append(f"unknown feature component: {name}")

    validate_augmentation(data, errors)
    validate_cleaning(data, errors)
    validate_normalization(data, errors)
    validate_quality_score(data, errors)

    name = model_name(data.get("model"))
    if name not in registry.MODELS:
        errors.append(f"unknown model component: {name}")
    model = data.get("model")
    if name == "score_fusion":
        validate_score_fusion_model(data, model, route_id, errors)
    elif isinstance(model, dict):
        for field in ["score_node", "components"]:
            if field in model:
                errors.append(f"model.{field} is only allowed for score_fusion routes")
        validate_model_specific_config(name, model, errors)

    adaptation = nested_name(data.get("adaptation")) or "none"
    if adaptation not in registry.ADAPTATION:
        errors.append(f"unknown adaptation component: {adaptation}")

    validate_training(data, errors)

    evaluation = data.get("evaluation")
    protocol = None
    metric = None
    if isinstance(evaluation, dict):
        protocol = evaluation.get("protocol")
        metric = evaluation.get("primary_metric")
    else:
        errors.append("evaluation must be a mapping")

    if not protocol:
        errors.append("evaluation.protocol is required")
    if not metric:
        errors.append("evaluation.primary_metric is required")
    if protocol not in registry.EVALUATION_PROTOCOLS:
        errors.append(f"unknown evaluation protocol: {protocol}")
    if metric is not None and metric not in registry.PRIMARY_METRICS:
        errors.append(f"unknown primary metric: {metric}")

    inference = data.get("inference")
    top4 = None
    if not isinstance(inference, dict):
        errors.append("inference must be a mapping")
    else:
        top4 = inference.get("top4")
        if "top4" not in inference:
            errors.append("inference.top4 is required")
        if "crop_policy" not in inference:
            errors.append("inference.crop_policy is required")

    if metric == "top4_BA" and top4 is not True:
        errors.append("top4_BA requires inference.top4: true")
    if metric == "no_top4_BA" and top4 is not False:
        errors.append("no_top4_BA requires inference.top4: false")
    if metric in {"exact_single_crop_expected_BA", "all_correct_rate"} and top4 is not True:
        errors.append(f"{metric} requires inference.top4: true")

    for required in ["dataset_version", "split_id"]:
        if not data.get(required):
            errors.append(f"{required} must be set")

    dataset_version = data.get("dataset_version")
    if isinstance(dataset_version, str) and dataset_version:
        dataset_path = registry.ROOT / "configs" / "datasets" / f"{dataset_version}.yaml"
        if not dataset_path.exists():
            errors.append(f"dataset_version has no manifest: {dataset_version}")

    split_id = data.get("split_id")
    if isinstance(split_id, str) and split_id:
        split_path = registry.ROOT / "configs" / "splits" / f"{split_id}.yaml"
        if not split_path.exists():
            errors.append(f"split_id has no manifest: {split_id}")

    if "tier" in data:
        errors.append("tier is not allowed; all routes are peers before audit")
    if "output_dir" in data:
        errors.append("output_dir is derived from route_id/run_id and must not be set in route config")

    return errors


def validate_positive_number(value: Any, field: str, errors: list[str]) -> float | None:
    if not isinstance(value, (int, float)) or value <= 0:
        errors.append(f"{field} must be positive")
        return None
    return float(value)


def validate_augmentation(data: dict[str, Any], errors: list[str]) -> None:
    augmentation = data.get("augmentation")
    if augmentation is None:
        return
    if not isinstance(augmentation, dict):
        errors.append("augmentation must be a mapping")
        return

    name = nested_name(augmentation)
    if name not in registry.AUGMENTATION:
        errors.append(f"unknown augmentation component: {name}")
        return

    if name not in {"split_first_sliding_window", "split_first_fixed_crops"}:
        return

    if augmentation.get("split_first") is not True:
        errors.append(f"{name} requires split_first: true")

    source_trial_sec = validate_positive_number(augmentation.get("source_trial_sec"), "augmentation.source_trial_sec", errors)
    window_sec = validate_positive_number(augmentation.get("window_sec"), "augmentation.window_sec", errors)
    stride_sec = None
    n_crops = None
    if name == "split_first_sliding_window":
        stride_sec = validate_positive_number(augmentation.get("stride_sec"), "augmentation.stride_sec", errors)
    else:
        n_crops = validate_positive_int(augmentation.get("n_crops"), "augmentation.n_crops", errors)

    input_window_sec = data.get("input_window_sec")
    if window_sec is not None and isinstance(input_window_sec, (int, float)) and abs(float(input_window_sec) - window_sec) > 1e-9:
        errors.append("input_window_sec must match augmentation.window_sec")
    if source_trial_sec is not None and window_sec is not None and window_sec > source_trial_sec:
        errors.append("augmentation.window_sec must be <= augmentation.source_trial_sec")
    if source_trial_sec is not None and stride_sec is not None and stride_sec > source_trial_sec:
        errors.append("augmentation.stride_sec must be <= augmentation.source_trial_sec")
    if source_trial_sec is not None and window_sec is not None and n_crops is not None:
        if window_sec * n_crops > source_trial_sec + 1e-9:
            errors.append("augmentation.n_crops * augmentation.window_sec must be <= augmentation.source_trial_sec")

    inference = data.get("inference") if isinstance(data.get("inference"), dict) else {}
    evaluation = data.get("evaluation") if isinstance(data.get("evaluation"), dict) else {}
    requires_exact_metric = (
        inference.get("crop_policy") == "exact_single_crop"
        or evaluation.get("primary_metric") == "exact_single_crop_expected_BA"
    )
    requires_exact_five_crops = name == "split_first_fixed_crops" and requires_exact_metric
    if requires_exact_five_crops and n_crops is not None and n_crops != 5:
        errors.append("split_first_fixed_crops exact_single_crop routes require augmentation.n_crops == 5")
    if (
        name == "split_first_sliding_window"
        and requires_exact_metric
        and source_trial_sec is not None
        and window_sec is not None
        and stride_sec is not None
    ):
        n_windows = sliding_window_count(source_trial_sec, window_sec, stride_sec)
        if n_windows != 5:
            errors.append(
                "METRIC_SCORE_MATRIX_SHAPE_COMPATIBLE: exact_single_crop_expected_BA "
                f"requires exactly 5 sliding-window crops, got {n_windows}"
            )

    search_space = augmentation.get("search_space")
    if requires_exact_five_crops and isinstance(search_space, dict) and "n_crops" in search_space:
        crop_values = search_space.get("n_crops")
        if isinstance(crop_values, list) and any(value != 5 for value in crop_values):
            errors.append("split_first_fixed_crops exact_single_crop routes require augmentation.search_space.n_crops values to be 5")

    apply_to_splits = augmentation.get("apply_to_splits")
    if apply_to_splits is not None:
        valid_splits = {"train", "val", "test"}
        if not isinstance(apply_to_splits, list) or not apply_to_splits or any(split not in valid_splits for split in apply_to_splits):
            errors.append("augmentation.apply_to_splits must be a non-empty list drawn from train/val/test")

    if "train_random_crop" in augmentation and not isinstance(augmentation["train_random_crop"], bool):
        errors.append("augmentation.train_random_crop must be a boolean")
    if "random_offset" in augmentation and not isinstance(augmentation["random_offset"], bool):
        errors.append("augmentation.random_offset must be a boolean")
    if (augmentation.get("train_random_crop") or augmentation.get("random_offset")) and name != "split_first_fixed_crops":
        errors.append("augmentation train random crop is only supported for split_first_fixed_crops")
    if augmentation.get("random_offset"):
        offset_sec = validate_non_negative_number(
            augmentation.get("random_offset_sec", 0.0),
            "augmentation.random_offset_sec",
            errors,
        )
        if offset_sec is not None and window_sec is not None and offset_sec >= window_sec:
            errors.append("augmentation.random_offset_sec must be < augmentation.window_sec")

    validate_augmentation_transforms(augmentation, errors)

    validate_augmentation_search_space(data, augmentation, errors)

    aggregate = augmentation.get("aggregate_to_trial")
    if not isinstance(aggregate, dict):
        errors.append("augmentation.aggregate_to_trial must be a mapping")
        return
    allowed_methods = {"majority_vote", "mean_score"}
    if aggregate.get("method") not in allowed_methods:
        errors.append("augmentation.aggregate_to_trial.method must be majority_vote or mean_score")
    # tie_break is consumed only by majority_vote. mean_score accepts the field
    # for config shape consistency with the sliding-window route family.
    if aggregate.get("tie_break") not in {None, "mean_score", "lower", "higher"}:
        errors.append("augmentation.aggregate_to_trial.tie_break must be mean_score, lower, or higher")


def validate_augmentation_transforms(augmentation: dict[str, Any], errors: list[str]) -> None:
    transforms = augmentation.get("transforms")
    if transforms is None:
        return
    if not isinstance(transforms, list):
        errors.append("augmentation.transforms must be a list")
        return
    for idx, item in enumerate(transforms):
        field = f"augmentation.transforms[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{field} must be a mapping")
            continue
        name = item.get("name")
        if name not in registry.AUGMENTATION_TRANSFORMS:
            errors.append(f"unknown augmentation transform: {name}")
        validate_probability(item.get("prob", 1.0), f"{field}.prob", errors)
        splits = item.get("apply_to_splits", ["train"])
        if not isinstance(splits, list) or not splits or any(split not in {"train", "val", "test"} for split in splits):
            errors.append(f"{field}.apply_to_splits must be a non-empty list drawn from train/val/test")
        elif set(splits) != {"train"}:
            errors.append(f"{field}.apply_to_splits must be [train]; augmentation transforms cannot alter val/test evidence")

        if name == "gaussian_noise":
            validate_non_negative_number(item.get("std", 0.01), f"{field}.std", errors)
            if "std_ratio" in item:
                validate_non_negative_number(item.get("std_ratio"), f"{field}.std_ratio", errors)
        elif name == "amplitude_scale":
            scale_range = item.get("scale_range", [0.9, 1.1])
            if not isinstance(scale_range, list) or len(scale_range) != 2:
                errors.append(f"{field}.scale_range must be [low, high]")
            else:
                low = validate_positive_number(scale_range[0], f"{field}.scale_range[0]", errors)
                high = validate_positive_number(scale_range[1], f"{field}.scale_range[1]", errors)
                if low is not None and high is not None and low > high:
                    errors.append(f"{field}.scale_range must be sorted")
        elif name == "channel_dropout":
            p = validate_non_negative_number(item.get("p", 0.1), f"{field}.p", errors)
            if p is not None and p >= 1.0:
                errors.append(f"{field}.p must be < 1")
            if "max_drop_channels" in item:
                validate_positive_int(item.get("max_drop_channels"), f"{field}.max_drop_channels", errors)
            if "exclude_channels" in item:
                exclude = item.get("exclude_channels")
                if not isinstance(exclude, list) or any(not isinstance(ch, str) or not ch for ch in exclude):
                    errors.append(f"{field}.exclude_channels must be a list of channel names")
        elif name == "channel_noise":
            validate_positive_int(item.get("max_channels", 2), f"{field}.max_channels", errors)
            validate_non_negative_number(item.get("noise_std_ratio", 0.03), f"{field}.noise_std_ratio", errors)
            if "exclude_channels" in item:
                exclude = item.get("exclude_channels")
                if not isinstance(exclude, list) or any(not isinstance(ch, str) or not ch for ch in exclude):
                    errors.append(f"{field}.exclude_channels must be a list of channel names")
        elif name == "time_mask":
            max_width = validate_positive_int(item.get("max_width", 25), f"{field}.max_width", errors)
            window_samples = augmentation_window_samples(augmentation)
            if max_width is not None and window_samples is not None and max_width >= window_samples:
                errors.append(f"{field}.max_width must be < augmentation.window_sec * 250Hz")
            if "mask_ratio" in item:
                validate_open_probability(item.get("mask_ratio"), f"{field}.mask_ratio", errors)
        elif name == "smooth_time_mask":
            validate_open_probability(item.get("mask_ratio", 0.05), f"{field}.mask_ratio", errors)
            validate_probability(item.get("attenuation", 0.0), f"{field}.attenuation", errors)
            edge_ratio = validate_probability(item.get("edge_ratio", 0.2), f"{field}.edge_ratio", errors)
            if edge_ratio is not None and edge_ratio >= 0.5:
                errors.append(f"{field}.edge_ratio must be < 0.5")
        elif name == "time_shift":
            max_shift = validate_non_negative_number(item.get("max_shift", 12), f"{field}.max_shift", errors)
            window_samples = augmentation_window_samples(augmentation)
            if max_shift is not None and window_samples is not None and max_shift >= window_samples:
                errors.append(f"{field}.max_shift must be < augmentation.window_sec * 250Hz")
        elif name == "random_bandstop":
            sfreq = validate_positive_number(item.get("sfreq", DEFAULT_SFREQ), f"{field}.sfreq", errors)
            width_hz = validate_positive_number(item.get("width_hz", 1.0), f"{field}.width_hz", errors)
            validate_probability(item.get("attenuation", 0.0), f"{field}.attenuation", errors)
            freq_range = item.get("freq_range", [4.0, 40.0])
            if not isinstance(freq_range, list) or len(freq_range) != 2:
                errors.append(f"{field}.freq_range must be [low, high]")
            else:
                low = validate_non_negative_number(freq_range[0], f"{field}.freq_range[0]", errors)
                high = validate_positive_number(freq_range[1], f"{field}.freq_range[1]", errors)
                if low is not None and high is not None:
                    if low > high:
                        errors.append(f"{field}.freq_range must be sorted")
                    if sfreq is not None and high >= sfreq / 2.0:
                        errors.append(f"{field}.freq_range[1] must be below Nyquist")
                    if width_hz is not None and width_hz > max(high - low, 1e-9) + 2.0:
                        errors.append(f"{field}.width_hz is too wide for freq_range")


def validate_cleaning(data: dict[str, Any], errors: list[str]) -> None:
    cleaning = data.get("cleaning")
    if cleaning is None:
        return
    if not isinstance(cleaning, dict):
        errors.append("cleaning must be a mapping")
        return
    method = cleaning.get("method")
    if method != "per_channel_robust_clip":
        errors.append(f"unknown cleaning.method: {method}")
        return
    if cleaning.get("stats_source") != "train_fold_only":
        errors.append("cleaning.stats_source must be train_fold_only")
    clip_n_mad = validate_positive_number(
        cleaning.get("clip_n_mad", cleaning.get("n_mad", 8.0)),
        "cleaning.clip_n_mad",
        errors,
    )
    if clip_n_mad is not None and clip_n_mad < 3.0:
        errors.append("cleaning.clip_n_mad must be >= 3.0")
    if "eps" in cleaning:
        validate_positive_number(cleaning["eps"], "cleaning.eps", errors)
    validate_split_list(
        cleaning.get("apply_to_splits", cleaning.get("apply_to", ["train", "val", "test"])),
        "cleaning.apply_to_splits",
        errors,
    )


def validate_normalization(data: dict[str, Any], errors: list[str]) -> None:
    normalization = data.get("normalization")
    if normalization is None:
        return
    if not isinstance(normalization, dict):
        errors.append("normalization must be a mapping")
        return
    if normalization.get("source") != "train_fold_only":
        errors.append("normalization.source must be train_fold_only")
    if normalization.get("scope") != "per_channel":
        errors.append("normalization.scope must be per_channel")
    if normalization.get("method") not in {"mean_std", "median_MAD"}:
        errors.append(f"unknown normalization.method: {normalization.get('method')}")
    if "eps" in normalization:
        validate_positive_number(normalization["eps"], "normalization.eps", errors)
    validate_split_list(
        normalization.get("apply_to_splits", normalization.get("apply_to", ["train", "val", "test"])),
        "normalization.apply_to_splits",
        errors,
    )


def validate_quality_score(data: dict[str, Any], errors: list[str]) -> None:
    quality = data.get("quality_score")
    if quality is None:
        return
    if not isinstance(quality, dict):
        errors.append("quality_score must be a mapping")
        return
    if "enabled" in quality and not isinstance(quality["enabled"], bool):
        errors.append("quality_score.enabled must be boolean")
    output = quality.get("output", "window_quality.csv")
    if not isinstance(output, str) or not output or "/" in output or "\\" in output:
        errors.append("quality_score.output must be a simple file name")


def validate_split_list(value: Any, field: str, errors: list[str]) -> None:
    allowed = {"train", "val", "valid", "test"}
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        errors.append(f"{field} must be a list of split names")
        return
    for item in items:
        if item not in allowed:
            errors.append(f"{field} has unsupported split: {item}")


def augmentation_window_samples(augmentation: dict[str, Any]) -> int | None:
    window_sec = augmentation.get("window_sec")
    if not isinstance(window_sec, (int, float)) or window_sec <= 0:
        return None
    return int(round(float(window_sec) * DEFAULT_SFREQ))


def validate_preprocessing_spec(item: dict[str, Any], errors: list[str]) -> None:
    name = preprocessing_component_name(item)
    if name != "bandpass":
        extra = sorted(key for key in item if key != "name")
        if extra:
            errors.append(f"preprocessing.{name} does not accept parameters: {', '.join(extra)}")
        return
    low_hz = validate_positive_number(item.get("low_hz", 1.0), "preprocessing.bandpass.low_hz", errors)
    high_hz = validate_positive_number(item.get("high_hz", 45.0), "preprocessing.bandpass.high_hz", errors)
    order = validate_positive_int(item.get("order", 4), "preprocessing.bandpass.order", errors)
    if low_hz is not None and high_hz is not None:
        if low_hz >= high_hz:
            errors.append("preprocessing.bandpass.low_hz must be < high_hz")
        if high_hz >= DEFAULT_SFREQ / 2.0:
            errors.append("preprocessing.bandpass.high_hz must be < 125Hz")
    if order is not None and order > 12:
        errors.append("preprocessing.bandpass.order must be <= 12")


def validate_model_specific_config(name: str | None, model: dict[str, Any], errors: list[str]) -> None:
    if name in {"dgcnn", "dual_graph_conformer", "lggnet", "tsception"}:
        montage = model.get("channel_montage")
        if montage not in {"hust_30_a2"}:
            errors.append(f"model.channel_montage must be hust_30_a2 for {name} routes")


def sliding_window_count(source_trial_sec: float, window_sec: float, stride_sec: float) -> int:
    return int((source_trial_sec - window_sec) / stride_sec + 1e-9) + 1


def route_primary_metric(data: dict[str, Any]) -> str | None:
    evaluation = data.get("evaluation")
    if not isinstance(evaluation, dict):
        return None
    metric = evaluation.get("primary_metric")
    return str(metric) if metric is not None else None


def validate_augmentation_search_space(data: dict[str, Any], augmentation: dict[str, Any], errors: list[str]) -> None:
    search_space = augmentation.get("search_space")
    if search_space is None:
        return
    if not isinstance(search_space, dict):
        errors.append("augmentation.search_space must be a mapping")
        return

    if augmentation.get("name") == "split_first_sliding_window":
        allowed = {"source_trial_sec", "window_sec", "stride_sec"}
    elif augmentation.get("name") == "split_first_fixed_crops":
        allowed = {"source_trial_sec", "window_sec", "n_crops"}
    else:
        allowed = {"source_trial_sec", "window_sec", "stride_sec", "n_crops"}

    source_trial_sec_base = (
        float(augmentation["source_trial_sec"])
        if isinstance(augmentation.get("source_trial_sec"), (int, float))
        else None
    )
    parsed_search: dict[str, list[float | int]] = {}
    for key, values in search_space.items():
        if key not in allowed:
            errors.append(f"augmentation.search_space has unsupported field: {key}")
            continue
        if not isinstance(values, list) or not values:
            errors.append(f"augmentation.search_space.{key} must be a non-empty list")
            continue
        if key == "n_crops":
            parsed_int = [validate_positive_int(value, f"augmentation.search_space.{key}", errors) for value in values]
            parsed = [value for value in parsed_int if value is not None]
        else:
            parsed_number = [validate_positive_number(value, f"augmentation.search_space.{key}", errors) for value in values]
            parsed = [value for value in parsed_number if value is not None]
        parsed_search[key] = parsed
        if key in {"window_sec", "stride_sec"} and source_trial_sec_base is not None:
            if any(float(value) > source_trial_sec_base for value in parsed):
                errors.append(f"augmentation.search_space.{key} values must be <= augmentation.source_trial_sec")
        if key == "n_crops" and source_trial_sec_base is not None and isinstance(augmentation.get("window_sec"), (int, float)):
            window_sec = float(augmentation["window_sec"])
            if any(float(value) * window_sec > source_trial_sec_base + 1e-9 for value in parsed):
                errors.append("augmentation.search_space.n_crops values must fit inside augmentation.source_trial_sec")

    def numeric_candidates(key: str) -> list[float] | None:
        base_value = augmentation.get(key)
        if not isinstance(base_value, (int, float)) or base_value <= 0:
            errors.append(
                f"SEARCH_SPACE_CROSS_PRODUCT_VALID: augmentation.{key} must be a positive base value "
                "before validating augmentation.search_space"
            )
            return None
        candidates = [float(base_value)]
        candidates.extend(float(value) for value in parsed_search.get(key, []))
        return sorted(set(candidates))

    def integer_candidates(key: str) -> list[int] | None:
        base_value = augmentation.get(key)
        if not isinstance(base_value, int) or base_value <= 0:
            errors.append(
                f"SEARCH_SPACE_CROSS_PRODUCT_VALID: augmentation.{key} must be a positive base value "
                "before validating augmentation.search_space"
            )
            return None
        candidates = [base_value]
        candidates.extend(int(value) for value in parsed_search.get(key, []))
        return sorted(set(candidates))

    window_values = numeric_candidates("window_sec")
    input_window_sec = data.get("input_window_sec")
    if window_values is not None and isinstance(input_window_sec, (int, float)):
        if any(abs(float(input_window_sec) - value) > 1e-9 for value in window_values):
            errors.append(
                "SEARCH_SPACE_CROSS_PRODUCT_VALID: input_window_sec must match every "
                "augmentation.window_sec/search_space.window_sec candidate"
            )

    source_values = numeric_candidates("source_trial_sec")
    metric = route_primary_metric(data)
    requires_five_crops = metric == "exact_single_crop_expected_BA"

    if augmentation.get("name") == "split_first_sliding_window":
        stride_values = numeric_candidates("stride_sec")
        if source_values is None or window_values is None or stride_values is None:
            return
        for source_trial_sec, window_sec, stride_sec in product(source_values, window_values, stride_values):
            combo = (
                f"source_trial_sec={source_trial_sec:g}, "
                f"window_sec={window_sec:g}, stride_sec={stride_sec:g}"
            )
            if window_sec > source_trial_sec:
                errors.append(f"SEARCH_SPACE_CROSS_PRODUCT_VALID: window_sec must be <= source_trial_sec for {combo}")
                continue
            if stride_sec > source_trial_sec:
                errors.append(f"SEARCH_SPACE_CROSS_PRODUCT_VALID: stride_sec must be <= source_trial_sec for {combo}")
                continue
            n_windows = sliding_window_count(source_trial_sec, window_sec, stride_sec)
            if requires_five_crops and n_windows != 5:
                errors.append(
                    "METRIC_SCORE_MATRIX_SHAPE_COMPATIBLE: exact_single_crop_expected_BA "
                    f"requires exactly 5 sliding-window crops, got {n_windows} for {combo}"
                )
        return

    if augmentation.get("name") == "split_first_fixed_crops":
        crop_values = integer_candidates("n_crops")
        if source_values is None or window_values is None or crop_values is None:
            return
        for source_trial_sec, window_sec, n_crops in product(source_values, window_values, crop_values):
            combo = (
                f"source_trial_sec={source_trial_sec:g}, "
                f"window_sec={window_sec:g}, n_crops={n_crops}"
            )
            if window_sec * n_crops > source_trial_sec + 1e-9:
                errors.append(
                    "SEARCH_SPACE_CROSS_PRODUCT_VALID: augmentation.search_space fixed-crop "
                    f"combinations must fit inside augmentation.source_trial_sec for {combo}"
                )
            if requires_five_crops and n_crops != 5:
                errors.append(
                    "METRIC_SCORE_MATRIX_SHAPE_COMPATIBLE: exact_single_crop_expected_BA "
                    f"requires exactly 5 fixed crops, got {n_crops} for {combo}"
                )


def validate_non_negative_number(value: Any, field: str, errors: list[str]) -> float | None:
    if not isinstance(value, (int, float)) or value < 0:
        errors.append(f"{field} must be non-negative")
        return None
    return float(value)


def validate_probability(value: Any, field: str, errors: list[str]) -> float | None:
    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        errors.append(f"{field} must be in [0, 1]")
        return None
    return float(value)


def validate_open_probability(value: Any, field: str, errors: list[str]) -> float | None:
    if not isinstance(value, (int, float)) or not 0.0 < float(value) < 1.0:
        errors.append(f"{field} must be in (0, 1)")
        return None
    return float(value)


def validate_positive_int(value: Any, field: str, errors: list[str]) -> int | None:
    if not isinstance(value, int) or value <= 0:
        errors.append(f"{field} must be a positive integer")
        return None
    return value


def validate_training(data: dict[str, Any], errors: list[str]) -> None:
    training = data.get("training")
    if training is None:
        return
    if not isinstance(training, dict):
        errors.append("training must be a mapping")
        return

    trainer = training.get("trainer")
    if trainer not in registry.TRAINERS:
        errors.append(f"unknown trainer component: {trainer}")

    validate_positive_int(training.get("epochs"), "training.epochs", errors)
    validate_positive_int(training.get("batch_size"), "training.batch_size", errors)

    loss = training.get("loss")
    if loss not in registry.LOSSES:
        errors.append(f"unknown loss component: {loss}")

    grad_clip_norm = training.get("grad_clip_norm")
    if grad_clip_norm is not None:
        validate_positive_number(grad_clip_norm, "training.grad_clip_norm", errors)

    optimizer = training.get("optimizer")
    if not isinstance(optimizer, dict):
        errors.append("training.optimizer must be a mapping")
    else:
        optimizer_name = optimizer.get("name")
        if optimizer_name not in registry.OPTIMIZERS:
            errors.append(f"unknown optimizer component: {optimizer_name}")
        validate_positive_number(optimizer.get("lr"), "training.optimizer.lr", errors)
        weight_decay = optimizer.get("weight_decay")
        if weight_decay is not None:
            validate_non_negative_number(weight_decay, "training.optimizer.weight_decay", errors)

    early_stopping = training.get("early_stopping")
    if early_stopping is not None:
        if not isinstance(early_stopping, dict):
            errors.append("training.early_stopping must be a mapping")
        else:
            monitor = early_stopping.get("monitor", "val_loss")
            if monitor not in {"train_loss", "train_accuracy", "val_loss", "val_accuracy"}:
                errors.append("training.early_stopping.monitor is not supported")
            mode = early_stopping.get("mode", "min")
            if mode not in {"min", "max"}:
                errors.append("training.early_stopping.mode must be min or max")
            patience = early_stopping.get("patience")
            if not isinstance(patience, int) or patience < 0:
                errors.append("training.early_stopping.patience must be a non-negative integer")
            min_delta = early_stopping.get("min_delta")
            if min_delta is not None:
                validate_non_negative_number(min_delta, "training.early_stopping.min_delta", errors)


def validate_score_fusion_model(data: dict[str, Any], model: Any, route_id: Any, errors: list[str]) -> None:
    if not isinstance(model, dict):
        errors.append("score_fusion model must be a mapping with score_node and components")
        return

    score_node = model.get("score_node")
    if not isinstance(score_node, str) or not score_node:
        errors.append("score_fusion model.score_node is required")

    components = model.get("components")
    if not isinstance(components, list) or not components or not all(isinstance(item, str) and item for item in components):
        errors.append("score_fusion model.components must be a non-empty list of strings")
        return

    if not isinstance(route_id, str) or not route_id:
        return
    route = score_route_by_id(route_id)
    if route is None:
        errors.append(f"score_fusion route is not registered in CLEAN_SCORE_ROUTES: {route_id}")
        return
    if score_node != route.score_node:
        errors.append(f"score_fusion score_node must match registered node: {route.score_node}")
    if tuple(components) != route.components:
        errors.append(f"score_fusion components must match registered components: {list(route.components)}")
