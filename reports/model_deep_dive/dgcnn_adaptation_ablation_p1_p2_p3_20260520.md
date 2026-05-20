# DGCNN Adaptation Ablation P1/P2/P3 2026-05-20

## Scope

- Branch: `codex/dgcnn-adaptation-ablation-routes`
- Commit: `6ef202a5f136b38bf86a61846a516c206fe7e1f2`
- Remote run root: `outputs/remote_dgcnn_ablation_pr26_full_20260520_102129`
- Dataset: `train_v1`
- Primary metric: `exact_single_crop_expected_BA`
- Status note: route statuses remain unchanged. This report summarizes remote candidate-gate protocol evidence and does not promote routes.

## Completion

All three DGCNN adaptation routes completed P1, P2, and P3 protocol summaries with no missing artifacts.

| Route | P1 jobs | P2 jobs | P3 final jobs | P3 artifact jobs | Summary status |
|---|---:|---:|---:|---:|---|
| `sliding_ea_dgcnn_dann_lambda010_w6_s1` | 25 | 7 | 5 | 60 | COMPLETE |
| `sliding_ea_dgcnn_dann_lambda010_w8_s0p5` | 25 | 7 | 5 | 60 | COMPLETE |
| `sliding_ea_dgcnn_coral_lambda005_w6_s1` | 25 | 7 | 5 | 60 | COMPLETE |

## P1 Results

| Rank | Route | n | Mean | Std | Min | Max |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `sliding_ea_dgcnn_dann_lambda010_w8_s0p5` | 25 | 0.6607 | 0.0652 | 0.5140 | 0.8092 |
| 2 | `sliding_ea_dgcnn_dann_lambda010_w6_s1` | 25 | 0.6472 | 0.0557 | 0.5240 | 0.7911 |
| 3 | `sliding_ea_dgcnn_coral_lambda005_w6_s1` | 25 | 0.6359 | 0.0583 | 0.5147 | 0.7499 |

## P2 Results

| Rank | Route | n | Mean | Std | Min | Max |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `sliding_ea_dgcnn_coral_lambda005_w6_s1` | 7 | 0.7024 | 0.0728 | 0.5417 | 0.7917 |
| 2 | `sliding_ea_dgcnn_dann_lambda010_w8_s0p5` | 7 | 0.6815 | 0.0441 | 0.5833 | 0.7292 |
| 3 | `sliding_ea_dgcnn_dann_lambda010_w6_s1` | 7 | 0.6250 | 0.0695 | 0.4583 | 0.6667 |

## P3 Results

| Rank | Route | n | Mean | Std | Min | Max |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `sliding_ea_dgcnn_dann_lambda010_w8_s0p5` | 5 | 0.6563 | 0.0222 | 0.6277 | 0.6949 |
| 2 | `sliding_ea_dgcnn_dann_lambda010_w6_s1` | 5 | 0.6462 | 0.0323 | 0.6037 | 0.6981 |
| 3 | `sliding_ea_dgcnn_coral_lambda005_w6_s1` | 5 | 0.6425 | 0.0321 | 0.5976 | 0.6861 |

## Interpretation

`sliding_ea_dgcnn_dann_lambda010_w8_s0p5` is the strongest overall route in this ablation. It ranks first in P1 and P3, and remains second in P2 with lower variance than the CORAL route. It is the best DGCNN adaptation candidate to carry into score fusion or follow-up route design.

`sliding_ea_dgcnn_coral_lambda005_w6_s1` is the best P2 stress-test route, with the highest holdout/crop-policy mean in this batch. Its weaker P1 and P3 means make it less attractive as a standalone route, but useful as a robustness-diverse component for ensemble experiments.

The DANN 8 s / 0.5 s route reaches near-parity with the strongest prior sliding-window FBSTCNet P3 result while improving the P1 and P2 means in this evidence batch. That makes DGCNN adaptation worth keeping in the next route family rather than treating it as a one-off diagnostic.

## Follow-Up

- Use `sliding_ea_dgcnn_dann_lambda010_w8_s0p5` as the DGCNN component in the next score-fusion route.
- Keep `sliding_ea_dgcnn_coral_lambda005_w6_s1` available for robustness-biased fusion because its P2 mean is the best in this PR26 run.
- Do not promote any route from this report alone; route status changes should wait for the repo's candidate/promotion policy and explicit audit-backed decision.
