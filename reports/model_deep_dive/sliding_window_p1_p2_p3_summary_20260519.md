# Sliding-Window P1/P2/P3 Summary 2026-05-19

## Scope

- Branch: `codex/resumable-experiment-runner`
- Run roots:
  - `outputs/local_sliding_p1_20260519_committed`
  - `outputs/local_sliding_p2_20260519_committed`
  - `outputs/local_sliding_p3_20260519_committed`
- Dataset: `train_v1`
- Primary metric: `exact_single_crop_expected_BA`
- Status note: all route statuses remain unchanged. This report summarizes local candidate-gate evidence and does not promote routes.

## Completion

All five sliding-window routes completed P1, P2, and P3 protocol summaries with no missing artifacts.

| Route | P1 jobs | P2 jobs | P3 final jobs | P3 artifact jobs | Final audit result |
|---|---:|---:|---:|---:|---|
| `sliding_ea_fbstcnet_m_conn_w8_s0p5` | 25 | 7 | 5 | 15 | PASS 5/5 |
| `sliding_ea_fbstcnet_m_conn_w6_s1` | 25 | 7 | 5 | 45 | PASS 5/5 |
| `sliding_ea_dgcnn_w6_s1` | 25 | 7 | 5 | 15 | PASS 5/5 |
| `sliding_pure_deformer_lite_w6_s1` | 25 | 7 | 5 | 15 | PASS 5/5 |
| `sliding_ea_fbstcnet_m_conn_w4_s1p5` | 25 | 7 | 5 | 15 | PASS 5/5 |

## P1 Results

| Rank | Route | n | Mean | Std | Min | Max |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `sliding_ea_fbstcnet_m_conn_w8_s0p5` | 25 | 0.6448 | 0.0598 | 0.5474 | 0.8036 |
| 2 | `sliding_ea_fbstcnet_m_conn_w4_s1p5` | 25 | 0.6401 | 0.0584 | 0.5150 | 0.7888 |
| 3 | `sliding_ea_fbstcnet_m_conn_w6_s1` | 25 | 0.6387 | 0.0535 | 0.5592 | 0.7887 |
| 4 | `sliding_ea_dgcnn_w6_s1` | 25 | 0.6345 | 0.0555 | 0.5221 | 0.7512 |
| 5 | `sliding_pure_deformer_lite_w6_s1` | 25 | 0.6306 | 0.0512 | 0.5491 | 0.7532 |

## P2 Results

| Rank | Route | n | Mean | Std | Min | Max |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `sliding_ea_dgcnn_w6_s1` | 7 | 0.6786 | 0.0746 | 0.5000 | 0.7292 |
| 2 | `sliding_ea_fbstcnet_m_conn_w8_s0p5` | 7 | 0.6726 | 0.0309 | 0.6042 | 0.7083 |
| 3 | `sliding_ea_fbstcnet_m_conn_w6_s1` | 7 | 0.6488 | 0.0624 | 0.5000 | 0.6875 |
| 4 | `sliding_ea_fbstcnet_m_conn_w4_s1p5` | 7 | 0.6339 | 0.0911 | 0.4167 | 0.7083 |
| 5 | `sliding_pure_deformer_lite_w6_s1` | 7 | 0.6310 | 0.0745 | 0.4583 | 0.6875 |

## P3 Results

| Rank | Route | n | Mean | Std | Min | Max |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `sliding_ea_fbstcnet_m_conn_w8_s0p5` | 5 | 0.6578 | 0.0444 | 0.5721 | 0.6980 |
| 2 | `sliding_ea_fbstcnet_m_conn_w6_s1` | 5 | 0.6518 | 0.0404 | 0.5721 | 0.6828 |
| 3 | `sliding_ea_dgcnn_w6_s1` | 5 | 0.6369 | 0.0197 | 0.6106 | 0.6632 |
| 4 | `sliding_pure_deformer_lite_w6_s1` | 5 | 0.6249 | 0.0212 | 0.5974 | 0.6564 |
| 5 | `sliding_ea_fbstcnet_m_conn_w4_s1p5` | 5 | 0.6248 | 0.0257 | 0.5960 | 0.6660 |

## Interpretation

`sliding_ea_fbstcnet_m_conn_w8_s0p5` is the best overall sliding-window variant in P1 and P3, and is second in P2. The result is consistent enough to treat the 8 s / 0.5 s window shape as the preferred sliding-window setting for the current FBSTCNet-M connection-light family.

`sliding_ea_dgcnn_w6_s1` is strongest in P2 but not in P1/P3. That suggests DGCNN may be more robust under the selected holdout/crop stress test while not dominating repeated CV or nested selection.

`sliding_ea_fbstcnet_m_conn_w6_s1` remains competitive and has the broadest P3 candidate search in this run: 45 inner artifact jobs plus 5 final jobs. It is a useful fallback because its P3 mean is close to the 8 s variant.

## Follow-Up

- Use `sliding_ea_fbstcnet_m_conn_w8_s0p5` as the first sliding-window candidate for deeper comparison.
- Keep `sliding_ea_dgcnn_w6_s1` in the route mix for robustness and model-family diversity.
- Continue official pretrained CBraMod remote evidence separately; it should be compared against the scratch CBraMod route after P1/P2/P3 artifacts finish.
