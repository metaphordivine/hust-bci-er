# DGCNN Adaptation Score Fusion Seed42 Protocol Report

Date: 2026-05-20

Branch: `codex/route-adaptation-score-fusion`
Route: `dgcnn_adaptation_fbstcnet_srfnet_conformer_score_average`
Remote worktree: `/root/autodl-tmp/hust-bci-er-route-adaptation-score-fusion-20260520-1610`
Remote run root: `outputs/remote_adaptation_score_fusion_full_20260520_161029`

## Result

The auditable seed42-aligned score-fusion recovery run passed the candidate gate.

| scope | result |
|---|---:|
| score-fusion run | `score_fusion_seed42_aligned/dgcnn_adaptation_fbstcnet_srfnet_conformer_score_average` |
| source component manifests | 25 |
| unique source splits | 5 |
| prediction rows | 480 |
| primary metric | `exact_single_crop_expected_BA` |
| primary metric value | 0.6984882666666666 |
| candidate gate | PASS |

Reproduction:

```bash
python scripts/repo_doctor.py experiment \
  --route configs/routes/models/dgcnn_adaptation_fbstcnet_srfnet_conformer_score_average.yaml \
  --run outputs/remote_adaptation_score_fusion_full_20260520_161029/score_fusion_seed42_aligned/dgcnn_adaptation_fbstcnet_srfnet_conformer_score_average \
  --gate candidate
```

## Execution Notes

The route first trained the missing P1 component evidence for:

- `fixed_crop_ea_fbstcnet`
- `sliding_window_srfnet`
- `sliding_window_conformer_lite`

Those three components completed 75 P1 job audits total across seeds `42`, `123`, `456`, `789`, and `1024`.

The first full assembly attempt failed correctly because the reusable DGCNN adaptation components only had seed42/fold0-4 component score matrices, while the newly trained components had five seeds. The assembly code refused to align mismatched `seed/fold/subject_id/trial_id` keys:

```text
component score table keys do not match first component: fixed_crop_ea_fbstcnet_component
```

The recovery run therefore used only the seed42/fold0-4 subset from all five components:

- `dgcnn_dann_cohort_component`
- `dgcnn_coral_cohort_component`
- `fixed_crop_ea_fbstcnet_component`
- `srfnet_long_component`
- `conformer_component`

This keeps the evidence aligned on the same folds without pretending that DANN/CORAL have complete five-seed coverage.

## Interpretation

The seed42-aligned fusion is valid candidate evidence for the registered score-fusion route, but it should not be read as a complete five-seed repeated P1 fusion sweep. The next stronger follow-up is to generate full P1 repeated DANN/CORAL component evidence, then rerun the five-component fusion over all seeds.

Route status remains `IDEA`; no route promotion was made.
