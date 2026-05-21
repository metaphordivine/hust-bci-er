# Interpretable Calibrated Diverse Score Fusion P1 Report

Date: 2026-05-20

Branch: `codex/interpretable-calibrated-fusion`
Route: `interpretable_calibrated_diverse_score_fusion`
Remote worktree: `/root/autodl-tmp/hust-bci-er-interpretable-calibrated-fusion-20260520-1910`
Remote run root: `outputs/remote_interpretable_calibrated_fusion_20260520_1910`

## Result

The calibrated diverse score-fusion route assembled successfully from genuine component score matrices and passed candidate audit after adding the route summary.

| scope | result |
|---|---:|
| source components | 6 |
| source manifests | 150 |
| unique source splits | 25 |
| score matrix evidence | `genuine` |
| primary metric | `exact_single_crop_expected_BA` |
| primary metric value | 0.6977498346666666 |
| candidate gate | PASS |

Reproduction:

```bash
python scripts/repo_doctor.py experiment \
  --route configs/routes/models/interpretable_calibrated_diverse_score_fusion.yaml \
  --run outputs/remote_interpretable_calibrated_fusion_20260520_1910/interpretable_calibrated_diverse_score_fusion \
  --gate candidate
```

## Components

| component | base route | source count | evidence |
|---|---|---:|---|
| `fixed_crop_ea_fbstcnet_component` | `fixed_crop_ea_fbstcnet` | 25 | genuine |
| `srfnet_long_component` | `sliding_window_srfnet` | 25 | genuine |
| `conformer_component` | `sliding_window_conformer_lite` | 25 | genuine |
| `dgcnn_dann_cohort_component` | `sliding_ea_dgcnn_dann_cohort_w6_s1` | 25 | genuine |
| `dgcnn_coral_cohort_component` | `sliding_ea_dgcnn_coral_cohort_w6_s1` | 25 | genuine |
| `riemannian_tangent_component` | `fixed_crop_ea_riemannian_tangent` | 25 | genuine |

All component tables used seeds `42`, `123`, `456`, `789`, and `1024`, folds `0` through `4`, and aligned through the same `seed/fold/subject_id/trial_id/crop_id` keys before fusion.

## Interpretation

The route is a stronger candidate than the standalone Riemannian tangent baseline, and it uses a broader family mix than the earlier seed42-aligned DGCNN adaptation fusion. The score is very close to the previous seed42-aligned route, but not clearly better:

- previous `dgcnn_adaptation_fbstcnet_srfnet_conformer_score_average` seed42-aligned score: 0.6984882666666666
- current full 25-split calibrated diverse score: 0.6977498346666666

This should be treated as valid candidate evidence and a useful fusion baseline, not as promotion evidence. The next meaningful step is P2 crop-stress evaluation and P3 inner-fold weight/temperature selection, rather than manually tuning weights against this P1 result.

## Explainability Hook

The route preserves component identity in its manifest and component score inputs. That lets follow-up diagnostics compute component correlation, disagreement, calibration, and explanation-stability weights without using subject or trial identifiers as model features.
