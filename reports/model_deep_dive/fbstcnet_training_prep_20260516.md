# FBSTCNet Training Prep 2026-05-16

This note prepares the first real-training batch after the FBSTCNet deep-dive
scaffold merge. It is planning and smoke evidence only; it does not promote any
route and it does not replace candidate audit evidence.

## Scope

Selected repository skills:

- Foundation Usage Skill: Real HUST EEG candidate run foundation.
- Smoke Audit Skill: route-only and bounded real-training smoke.
- Model Smoke Skill: Torch backbone forward validation.

## Preflight Evidence

- Branch base: merged `origin/main` after PR 14.
- Data root checked: `scratch/local_data/hust_bci_er_train/训练集`.
- Runtime checked: `torch` is available with CUDA; `h5py` is available.
- `python scripts/repo_doctor.py fast`: PASS, `283 passed`.
- `python scripts/model_smoke.py`: PASS for `cbramod`, `conformer_lite`,
  `deformer_lite`, `eegnet`, `fbstcnet`, `shallow_conv_net`, and `srfnet`.

Route-only smoke checks completed:

| route | gate | result |
|---|---|---|
| `fixed_crop_ea_fbstcnet` | smoke | WARN allowed |
| `fixed_crop_fbstcnet_zscore_only` | smoke | WARN allowed |
| `fixed_crop_ea_fbstcnet_p_only` | smoke | WARN allowed |
| `fixed_crop_ea_fbstcnet_bands_4_40` | smoke | WARN allowed |

The WARN state is expected for route-only smoke because no finished run
artifact is provided. Route statuses remain unchanged.

## Bounded Real-Training Smoke

The first new ablation route was run through a one-epoch real-data smoke:

```bash
python scripts/launch_reproducible.py --seed 42 -- \
  python scripts/train_route.py \
    --mode smoke \
    --route configs/routes/models/fixed_crop_fbstcnet_zscore_only.yaml \
    --run-dir outputs/training_prep/fixed_crop_fbstcnet_zscore_only_smoke_20260516_1ep_repro \
    --smoke-epochs 1 \
    --device auto \
    --data-root scratch/local_data/hust_bci_er_train/训练集
```

Output:

```text
exact_single_crop_expected_BA: 0.6052296
run_dir: outputs/training_prep/fixed_crop_fbstcnet_zscore_only_smoke_20260516_1ep_repro
```

Smoke audit command:

```bash
python scripts/repo_doctor.py experiment \
  --route configs/routes/models/fixed_crop_fbstcnet_zscore_only.yaml \
  --run outputs/training_prep/fixed_crop_fbstcnet_zscore_only_smoke_20260516_1ep_repro \
  --gate smoke
```

Result: `overall: WARN`, which is acceptable for smoke-mode evidence and must
not be interpreted as candidate readiness.

## Candidate Training Queue

Run full candidate jobs with route default epochs and no `--epochs-override`.
Use the reproducible launcher so `PYTHONHASHSEED` and deterministic settings are
locked.

First batch:

1. `fixed_crop_ea_fbstcnet`
2. `fixed_crop_fbstcnet_zscore_only`
3. `fixed_crop_ea_fbstcnet_p_only`
4. `fixed_crop_ea_fbstcnet_c_only`
5. `fixed_crop_ea_fbstcnet_bands_4_40`
6. `fixed_crop_ea_fbstcnet_f1_24_f2_48`

Template:

```bash
python scripts/launch_reproducible.py --seed 42 -- \
  python scripts/run_candidate_route.py \
    --route configs/routes/models/<route_id>.yaml \
    --run-dir outputs/<route_id>/candidate_20260516_seed42 \
    --device auto \
    --data-root scratch/local_data/hust_bci_er_train/训练集
```

After each finished run:

```bash
python scripts/repo_doctor.py experiment \
  --route configs/routes/models/<route_id>.yaml \
  --run outputs/<route_id>/candidate_20260516_seed42 \
  --gate candidate
```

If candidate audit passes, record only the concise route summary under
`reports/route_summaries/`. Do not commit checkpoints, raw outputs, full
prediction tables, or temporary experiment dumps.

## Score-Fusion Dependency

The FBSTCNet score-fusion routes should wait until the single-model component
score matrices above are available as traceable compact evidence. Do not run or
promote score-fusion candidate routes while their blocker remains
`missing_component_evidence`.
