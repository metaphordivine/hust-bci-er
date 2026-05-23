# DEP/HC Router Covariance-Tangent Diagnostic 2026-05-23

This is a separate DEP/HC subject-kind diagnostic task, not an emotion Top-4
route and not candidate evidence.

## Scope

- Model: covariance tangent features + log bandpower features + logistic router.
- Input: fixed 10s EEG crops.
- Output: `p_dep`, `p_hc`, `confidence`, subject-level aggregate prediction.
- Guardrail: `subject_id` is used only for split membership, aggregation, and
  reporting. It is not a model input feature.
- Split policy: subject-level P1/P2 split helpers from the repository.

## Local Smoke Evidence

Local data root: `scratch/local_data/hust_bci_er_train/训练集`.

| run | output root | subject BA | DEP recall | HC recall | notes |
|---|---|---:|---:|---:|---|
| P1 seed42 fold0 | `outputs/dep_hc_router_p1_seed42_fold0_calibrated_local` | 0.8125 | 1.0000 | 0.6250 | Strong signal, DEP-biased. |
| P2 holdout999 | `outputs/dep_hc_router_p2_holdout999_calibrated_local` | 0.6250 | 0.7500 | 0.5000 | Generalizes weakly; not ready for hard routing. |

## Interpretation

The router can learn a DEP/HC signal on held-out P1 subjects, but P2 shows that
the signal is not yet stable enough to drive expert selection by itself. The
next useful work is to add traditional feature ablations such as PSD,
frontal-alpha asymmetry, entropy/connectivity, and a small EEGNet-lite router,
then compare subject-level calibration and DEP/HC recall before any downstream
expert routing.
