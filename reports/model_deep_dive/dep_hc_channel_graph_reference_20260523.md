# DEP/HC Channel Graph Reference

This note records the channel-order evidence used by the independent DEP/HC
classifier graph feature branch. It is reference material only and does not
change any split, route status, or candidate evidence.

Source PDF: `scratch/local_data/hust_bci_er_train/说明文档/数据集说明文档f.pdf`.

Key dataset facts from the PDF:

- Training subjects: 40 HC and 20 DEP.
- Test subjects: 5 HC and 5 DEP. Public test labels are not available and must
  not be used for training or evaluation.
- Sampling rate: 250 Hz.
- Reference: A2.
- Each training subject has four neutral and four positive emotion videos,
  approximately 60 seconds each.
- Training arrays are shaped as 30 channels by 50000 samples for each emotion
  group. The original emotion labels remain positive=1 and neutral=0; DEP/HC is
  a separate subject-kind target.

HUST 30-channel A2 order:

| Index | Channel | Region prior |
|---:|---|---|
| 1 | FP1 | frontal |
| 2 | FP2 | frontal |
| 3 | F7 | frontal |
| 4 | F3 | frontal |
| 5 | FZ | frontal |
| 6 | F4 | frontal |
| 7 | F8 | frontal |
| 8 | FT7 | frontocentral |
| 9 | FC3 | frontocentral |
| 10 | FCZ | frontocentral |
| 11 | FC4 | frontocentral |
| 12 | FT8 | frontocentral |
| 13 | T3 | central |
| 14 | C3 | central |
| 15 | CZ | central |
| 16 | C4 | central |
| 17 | T4 | central |
| 18 | TP7 | centroparietal |
| 19 | CP3 | centroparietal |
| 20 | CPZ | centroparietal |
| 21 | CP4 | centroparietal |
| 22 | TP8 | centroparietal |
| 23 | T5 | posterior |
| 24 | P3 | posterior |
| 25 | PZ | posterior |
| 26 | P4 | posterior |
| 27 | T6 | posterior |
| 28 | O1 | posterior |
| 29 | OZ | posterior |
| 30 | O2 | posterior |

The first graph prior uses two conservative edge families:

- local region-chain edges between adjacent channels within each region;
- left/right homologous edges such as FP1-FP2, F3-F4, C3-C4, P3-P4, O1-O2.

These priors are intended for low-capacity DEP/HC graph diagnostics and
feature-level regional connectivity summaries. They should not be treated as
proof that a graph model is valid; they are only a leakage-free structural
prior derived from dataset channel semantics.
