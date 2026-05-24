# Component Registry

## Preprocessing

- `bandpass`
- `car`
- `euclidean_alignment`
- `robust_zscore`
- `shrinkage_whitening`
- `whitening_eps1e3`
- `whitening_eps3e4`
- `zscore`

## Features

- `bandpower`
- `connectivity`
- `differential_entropy`
- `hjorth`
- `riemannian_tangent`

## Augmentation

- `split_first_fixed_crops`
- `split_first_sliding_window`

## Augmentation Transforms

- `amplitude_scale`
- `band_amplitude_scale`
- `channel_dropout`
- `channel_noise`
- `dc_shift`
- `gaussian_noise`
- `random_bandstop`
- `region_scale_down`
- `smooth_time_mask`
- `time_mask`
- `time_shift`

## Torch Backbones

- `cbramod`
- `cbramod_pretrained`
- `conformer_lite`
- `deformer_lite`
- `eegnet`
- `fbcnet`
- `fbstcnet`
- `riemannian_tangent`
- `shallow_conv_net`
- `srf_fbstcnet_gate`
- `srfnet`
- `tri_context_gate`
- `tsception`

## Sklearn Models

- `extra_trees`
- `logistic_regression`
- `random_forest`

## Graph Models

- `dgcnn`
- `dual_graph_conformer`
- `lggnet`

## Score Route Models

- `score_fusion`

## Adaptation

- `adabn`
- `coral`
- `dann`
- `masked_consistency`
- `none`

## Evaluation Protocols

- `p1_repeated_group_kfold`
- `p2_pseudo_public_holdout`
- `p3_nested_selection`

## Primary Metrics

- `all_correct_rate`
- `exact_single_crop_expected_BA`
- `no_top4_BA`
- `top4_BA`
