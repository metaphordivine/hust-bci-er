# Lightweight Existing Evidence Evaluation

This report is generated from committed route summaries, committed compact protocol board snapshots, and explicitly provided COMPLETE protocol boards. It does not read raw EEG data, train models, or promote routes.

## Coverage

- P1: 7 routes with any evidence (7 protocol-board, 0 route-summary-only)
- P2: 7 routes with any evidence (7 protocol-board, 0 route-summary-only)
- P3: 7 routes with any evidence (7 protocol-board, 0 route-summary-only)

## Protocol Leaders

### P1

| Route | Family | Model | Status | Mean | Jobs | Source |
|---|---|---|---|---|---|---|
| `sliding_ea_fbstcnet_m_conn_w8_s0p5` | filterbank_cnn | fbstcnet | IDEA | 0.6448 | 25 | protocol_board |
| `sliding_ea_fbstcnet_m_conn_w4_s1p5` | filterbank_cnn | fbstcnet | IDEA | 0.6401 | 25 | protocol_board |
| `sliding_ea_fbstcnet_m_conn_w6_s1` | filterbank_cnn | fbstcnet | IDEA | 0.6387 | 25 | protocol_board |
| `sliding_ea_dgcnn_w6_s1` | graph | dgcnn | IDEA | 0.6345 | 25 | protocol_board |
| `sliding_pure_deformer_lite_w6_s1` | attention_hybrid | deformer_lite | IDEA | 0.6306 | 25 | protocol_board |
| `sliding_ea_cbramod_pretrained_w3p2_s1p7` | cbramod | cbramod_pretrained | IDEA | 0.6285 | 25 | protocol_board |
| `tuned_sliding_window_cbramod` | cbramod | cbramod | CANDIDATE | 0.5773 | 25 | protocol_board |

### P2

| Route | Family | Model | Status | Mean | Jobs | Source |
|---|---|---|---|---|---|---|
| `sliding_ea_dgcnn_w6_s1` | graph | dgcnn | IDEA | 0.6786 | 7 | protocol_board |
| `sliding_ea_fbstcnet_m_conn_w8_s0p5` | filterbank_cnn | fbstcnet | IDEA | 0.6726 | 7 | protocol_board |
| `sliding_ea_fbstcnet_m_conn_w6_s1` | filterbank_cnn | fbstcnet | IDEA | 0.6488 | 7 | protocol_board |
| `sliding_ea_fbstcnet_m_conn_w4_s1p5` | filterbank_cnn | fbstcnet | IDEA | 0.6339 | 7 | protocol_board |
| `sliding_pure_deformer_lite_w6_s1` | attention_hybrid | deformer_lite | IDEA | 0.6310 | 7 | protocol_board |
| `sliding_ea_cbramod_pretrained_w3p2_s1p7` | cbramod | cbramod_pretrained | IDEA | 0.5833 | 7 | protocol_board |
| `tuned_sliding_window_cbramod` | cbramod | cbramod | CANDIDATE | 0.5357 | 7 | protocol_board |

### P3

| Route | Family | Model | Status | Mean | Jobs | Source |
|---|---|---|---|---|---|---|
| `sliding_ea_fbstcnet_m_conn_w8_s0p5` | filterbank_cnn | fbstcnet | IDEA | 0.6578 | 5 | protocol_board |
| `sliding_ea_fbstcnet_m_conn_w6_s1` | filterbank_cnn | fbstcnet | IDEA | 0.6518 | 5 | protocol_board |
| `sliding_ea_cbramod_pretrained_w3p2_s1p7` | cbramod | cbramod_pretrained | IDEA | 0.6408 | 5 | protocol_board |
| `sliding_ea_dgcnn_w6_s1` | graph | dgcnn | IDEA | 0.6369 | 5 | protocol_board |
| `sliding_pure_deformer_lite_w6_s1` | attention_hybrid | deformer_lite | IDEA | 0.6249 | 5 | protocol_board |
| `sliding_ea_fbstcnet_m_conn_w4_s1p5` | filterbank_cnn | fbstcnet | IDEA | 0.6248 | 5 | protocol_board |
| `tuned_sliding_window_cbramod` | cbramod | cbramod | CANDIDATE | 0.5629 | 5 | protocol_board |

## Family Leaders By P3

| Route | Family | Model | P3 Mean |
|---|---|---|---|
| `sliding_ea_fbstcnet_m_conn_w8_s0p5` | filterbank_cnn | fbstcnet | 0.6578 |
| `sliding_ea_cbramod_pretrained_w3p2_s1p7` | cbramod | cbramod_pretrained | 0.6408 |
| `sliding_ea_dgcnn_w6_s1` | graph | dgcnn | 0.6369 |
| `sliding_pure_deformer_lite_w6_s1` | attention_hybrid | deformer_lite | 0.6249 |

## P3 Minus P1 Gap

| Route | Family | P1 | P3 | P3-P1 |
|---|---|---|---|---|
| `sliding_ea_fbstcnet_m_conn_w8_s0p5` | filterbank_cnn | 0.6448 | 0.6578 | +0.0130 |
| `sliding_ea_fbstcnet_m_conn_w6_s1` | filterbank_cnn | 0.6387 | 0.6518 | +0.0131 |
| `sliding_ea_cbramod_pretrained_w3p2_s1p7` | cbramod | 0.6285 | 0.6408 | +0.0123 |
| `sliding_ea_dgcnn_w6_s1` | graph | 0.6345 | 0.6369 | +0.0024 |
| `sliding_pure_deformer_lite_w6_s1` | attention_hybrid | 0.6306 | 0.6249 | -0.0057 |
| `sliding_ea_fbstcnet_m_conn_w4_s1p5` | filterbank_cnn | 0.6401 | 0.6248 | -0.0153 |
| `tuned_sliding_window_cbramod` | cbramod | 0.5773 | 0.5629 | -0.0144 |

## Use

- Treat this as triage evidence only; it is not a route promotion artifact.
- Prefer candidates that combine high P3 score with family diversity.
- Large negative P3-P1 gaps are follow-up diagnostics for selection robustness or cohort/session shift.
