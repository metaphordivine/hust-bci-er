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

- `channel_dropout`
- `gaussian_noise`
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
- `shallow_conv_net`
- `srfnet`
- `tsception`

## Sklearn Models

- `extra_trees`
- `logistic_regression`
- `random_forest`

## Graph Models

- `dgcnn`
- `lggnet`

## Score Route Models

- `score_fusion`

## Adaptation

- `adabn`
- `dann`
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
