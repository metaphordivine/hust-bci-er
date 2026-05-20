# CBraMod Pretrained vs Scratch P1/P2/P3 2026-05-19

## Scope

- Branch: `codex/resumable-experiment-runner`
- Dataset: `train_v1`
- Primary metric: `exact_single_crop_expected_BA`
- Official pretrained route: `sliding_ea_cbramod_pretrained_w3p2_s1p7`
- Scratch comparison route: `tuned_sliding_window_cbramod`
- Status note: route statuses remain unchanged. This report compares completed protocol evidence and does not promote routes.

## Completion

Both CBraMod-family routes completed P1, P2, and P3 with candidate-gate artifacts.

| Route | P1 jobs | P2 jobs | P3 final jobs | P3 artifact jobs | Final audit result |
|---|---:|---:|---:|---:|---|
| `sliding_ea_cbramod_pretrained_w3p2_s1p7` | 25 | 7 | 5 | 45 | PASS 5/5 |
| `tuned_sliding_window_cbramod` | 25 | 7 | 5 | 15 | PASS 5/5 |

## Results

| Route | Protocol | n | Mean | Std | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| `sliding_ea_cbramod_pretrained_w3p2_s1p7` | P1 | 25 | 0.6285 | 0.0373 | 0.5696 | 0.7131 |
| `tuned_sliding_window_cbramod` | P1 | 25 | 0.5773 | 0.0387 | 0.4964 | 0.6450 |
| `sliding_ea_cbramod_pretrained_w3p2_s1p7` | P2 | 7 | 0.5833 | 0.1303 | 0.2917 | 0.7292 |
| `tuned_sliding_window_cbramod` | P2 | 7 | 0.5357 | 0.0719 | 0.3750 | 0.6042 |
| `sliding_ea_cbramod_pretrained_w3p2_s1p7` | P3 | 5 | 0.6408 | 0.0280 | 0.6052 | 0.6866 |
| `tuned_sliding_window_cbramod` | P3 | 5 | 0.5629 | 0.0139 | 0.5428 | 0.5843 |

## Delta

| Protocol | Pretrained mean | Scratch mean | Delta |
|---|---:|---:|---:|
| P1 | 0.6285 | 0.5773 | +0.0511 |
| P2 | 0.5833 | 0.5357 | +0.0476 |
| P3 | 0.6408 | 0.5629 | +0.0779 |

## Interpretation

Official CBraMod pretrained finetuning is consistently better than the scratch CBraMod route across P1, P2, and P3. The largest gain appears in P3, where selected-artifact outer-test evaluation benefits by about 0.078 balanced accuracy.

The pretrained route is still not the top overall route from this campaign: the best sliding-window FBSTCNet-M connection variant remains higher in P1/P2/P3. The practical conclusion is to keep official CBraMod pretrained finetuning as a real foundation-model family for diversity and future ensemble or adaptation work, but not to treat it as the current single best competition route.

## Artifact Notes

- Local copied roots are under `outputs/autodl_cbramod_20260519/`.
- The official pretrained checkpoint is ignored at `scratch/model_weights/cbramod/pretrained_weights.pth` and is identified by SHA256 in the model provenance tests.
- Remote large outputs and checkpoints are not committed.
