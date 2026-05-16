# FBSTCNet Next Stage Plan

This plan is diagnostic and route-planning evidence only. It does not promote
any route and it does not replace `repo_doctor.py experiment` gates.

## Scope

The next stage should stop broad model sampling and focus on two lines:

1. Stabilize `fixed_crop_ea_fbstcnet`, the current strongest single model.
2. Add traceable score-fusion routes that combine the fixed-crop EA FBSTCNet
   component with SRFNet, Conformer, and whitening SRFNet components.

## Step 1: Diagnostics

Use the committed score-matrix diagnostic entry point before launching more
candidate runs:

```bash
python scripts/analyze_candidate_scores.py \
  --runs outputs/candidate_all_models/<run_id> outputs/model_route_exploration/<run_id> \
  --out reports/model_deep_dive/leaderboard.md
```

The report includes:

- `exact_single_crop_expected_BA` and mean-score Top-4 BA by route.
- DEP/HC cohort BA and per-crop Top-4 BA.
- Score calibration summaries.
- Pairwise score correlation and Top-4 disagreement matrices.
- Frequent Top-4 error samples across models.

## Step 2: FBSTCNet Ablations

The first ablation batch is capped at 12 IDEA routes and keeps one factor
changed per route where possible:

- Preprocessing: `fixed_crop_fbstcnet_zscore_only`,
  `fixed_crop_car_fbstcnet`, `fixed_crop_whitening_eps1e3_fbstcnet`,
  `fixed_crop_ea_car_fbstcnet`, `fixed_crop_ea_whitening_eps1e3_fbstcnet`.
- Branches: `fixed_crop_ea_fbstcnet_p_only`,
  `fixed_crop_ea_fbstcnet_c_only`,
  `fixed_crop_ea_fbstcnet_m_power_light`,
  `fixed_crop_ea_fbstcnet_m_conn_light`.
- Filterbank/capacity: `fixed_crop_ea_fbstcnet_fft_rectangular`,
  `fixed_crop_ea_fbstcnet_bands_4_40`,
  `fixed_crop_ea_fbstcnet_f1_24_f2_48`.

These routes stay `IDEA` until a real run produces manifest, prediction,
score-matrix, metric report, audit report, and route summary evidence.

## Step 3: Score Fusion

The following IDEA score-fusion routes are registered:

- `fbstcnet_srfnet_score_average`: 0.50 FBSTCNet + 0.50 SRFNet.
- `fbstcnet_srfnet_conformer_score_average`: 0.40 FBSTCNet +
  0.40 SRFNet + 0.20 Conformer.
- `fbstcnet_srfnet_whitening_eps1e3_average`: 0.40 FBSTCNet +
  0.30 SRFNet reference + 0.30 whitening SRFNet eps1e3.
- `fbstcnet_as_query_srfnet_context_fusion`: FBSTCNet query with
  SRFNet/Conformer context.
- `srfnet_as_query_fbstcnet_context_fusion`: SRFNet query with
  FBSTCNet/Conformer context.

For score-fusion configs, `preprocessing` is component provenance only. It does
not mean the assembler reprocesses every component through one shared
preprocessing pipeline.

Before any score-fusion route can become `CANDIDATE`, its component score
matrices must be available as traceable compact evidence, not only as local
`outputs/...` paths.

## Step 4: Controlled Side Routes

Deformer is limited to eight controlled IDEA routes:

- `fixed_crop_ea_deformer_mean_pool_control`
- `fixed_crop_ea_deformer_depth1_emb64`
- `fixed_crop_ea_deformer_depth3_emb64`
- `fixed_crop_ea_deformer_depth2_emb96`
- `fixed_crop_ea_deformer_dropout015`
- `fixed_crop_ea_deformer_dropout035`
- `fixed_crop_ea_deformer_ff2`
- `fixed_crop_ea_deformer_ff6`

CBraMod is limited to rescue validation routes:

- `fixed_crop_ea_cbramod_candidate`
- `fixed_crop_ea_cbramod_patch125`
- `fixed_crop_ea_cbramod_patch500`
- `tuned_sliding_window_cbramod_source14_stride2`

`fixed_crop_ea_cbramod_patch125` uses explicit `conv_out_channels` and
`group_norm_groups` so the patch embedding width remains compatible with
`d_model`.

## Step 5: Promotion Gate

Single-split route decisions should use these thresholds only as triage:

- New single-model champion: `exact_single_crop_expected_BA > 0.690`.
- Fusion component: `> 0.620` plus low correlation or useful error
  disagreement with the champion.
- Stop route: `< 0.580` and no clear complementary error pattern.

For any final claim beyond a single split, run repeated group-kfold or
multi-seed validation through the repository protocol tooling, then require:

```bash
python scripts/repo_doctor.py fast
python scripts/repo_doctor.py experiment --route <route_config> --run <run_dir> --gate candidate
```

No route in this plan should be changed to `CANDIDATE` or `PROMOTED` unless the
corresponding audit report supports that lifecycle change.

