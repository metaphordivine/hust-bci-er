# Foundation Usage Protocol

This protocol is platform-neutral. It tells agents how to compose the reusable
foundation modules in this repository instead of creating one-off scripts or
bypassing audit contracts.

## Trigger

Use this protocol when the task says or implies any of the following:

- build or refresh dataset evidence
- build or refresh subject-level split evidence
- run P1/P2/P3 through the unified runner
- write predictions, component scores, score-route outputs, or metric reports
- record a reproducible training or evaluation run
- prepare route summaries, promotion audits, registry docs, QA reports, caches, or monitors
- make an agent workflow more reliable before a real experiment

This protocol does not replace the experiment audit gate. It routes work to the
right foundation pieces, then calls the repository gates.

## Foundation Map

| Scenario | Use this foundation | Stable entry point | Expected artifact | Completion gate |
|---|---|---|---|---|
| Raw/indexed data becomes evidence | Dataset Manifest Builder | `scripts/build_dataset_manifest.py` / `hust_bci_er.data.manifest_builder` | `configs/datasets/<dataset_version>.yaml` with data sources, checksums, subject/trial/crop rows, label availability | Dataset QA plus `python scripts/repo_doctor.py fast` |
| Subject split becomes evidence | Split Manifest Builder | `scripts/build_split_manifest.py` / `hust_bci_er.data.split_builder` | `configs/splits/<split_id>.yaml` with subject lists and `trial_rows` | No subject/original-trial leakage plus fast gate |
| Dataset evidence needs sanity check | Lightweight Data QA | `scripts/data_qa_report.py` / `hust_bci_er.data.qa` | Markdown/JSON QA summary for subject, trial, crop, label, checksum gaps | QA report has no blocking gaps for the intended gate |
| P1/P2/P3 should become runnable jobs | Protocol Runner Skeleton | `scripts/plan_evaluation_protocol.py`, `scripts/run_evaluation_protocol.py` | `protocol_run_manifest.json` and job split contracts | Runner manifest exists; do not call it candidate evidence |
| Toy end-to-end platform smoke | Toy Route Job Adapter | `scripts/launch_reproducible.py`, `scripts/train_route.py`, `scripts/toy_experiment_audit.py` | synthetic dataset/split evidence, predictions, score matrix, metric report, run manifest, candidate audit report under `outputs/` | toy candidate audit passes; do not call it real EEG evidence |
| Real HUST EEG candidate run | Torch Classifier Route Job Adapter | `scripts/launch_reproducible.py`, `scripts/run_candidate_route.py` | run-local dataset/split evidence, held-out test predictions, genuine score matrix, metric report, manifest, route summary, candidate audit report | `repo_doctor.py experiment --gate candidate` passes |
| Real run needs reproducibility lock | Deterministic Utilities + Run Manifest Writer | `hust_bci_er.training.reproducibility`, `hust_bci_er.audit.run_manifest.write_run_manifest` | `manifest.json` with git commit, config snapshot hash, dataset/split hash, command, seed, environment, artifact hashes | `repo_doctor.py experiment --gate candidate` when run artifacts exist |
| Model/component output needs a contract | Component Artifact Contract | `hust_bci_er.contracts.artifacts` | component score CSV with `component_id`, subject/trial/crop metadata, score, optional `y_true` | Contract validator passes |
| Prediction output needs a contract | Prediction Writer | `hust_bci_er.evaluation.prediction_writer` | prediction CSV with `subject_id`, `trial_id`, optional `crop_id`, `score`, `pred_top4`, `y_true`/`y_pred` as appropriate | Audit can recompute the declared primary metric |
| Component scores need route assembly | Score Route Assembly | `scripts/assemble_score_route.py` / `hust_bci_er.inference.score_route_assembly` | score-fusion prediction CSV preserving alignment keys and labels when available | Score route tests and experiment audit pass |
| Prediction/score outputs need reports | Metric Report Builder | `hust_bci_er.evaluation.report` | metric board, subject metric rows, audit JSON | Report values match the declared primary metric |
| Audit evidence needs a concise route summary | Route Summary Generator | `scripts/generate_route_summary.py` / `hust_bci_er.audit.summary` | concise route summary under `reports/route_summaries/` | Summary consistency check passes |
| Source paths need leakage guard | No-Leakage Source Scanner | `scripts/scan_no_leakage.py` / `hust_bci_er.audit.source_scanner` | scanner findings or pass output | No public/private label, leaderboard, ID shortcut, inference-label leaks |
| Promotion evidence is requested | Promotion Audit Helper | `reports/promotion_audits/_template.md`, `scripts/check_promotion_audit.py` | promotion audit referencing a passing candidate audit report | promoted gate passes |
| Registry/docs drift is suspected | Registry Consistency + Docs Generator | `scripts/check_registry_consistency.py`, `scripts/generate_component_docs.py` | consistent registry/factory/config names; refreshed component docs | registry consistency plus fast gate |
| Long run needs progress tracking | Training Monitor | `hust_bci_er.training.monitor` | `training_events.jsonl`, `heartbeat.json` | monitor records epoch and finish/error events |
| Local derived artifacts need tracking | Cache Manager | `hust_bci_er.audit.cache_manager` | local cache manifest outside committed outputs | cache paths stay under cache dir; do not commit raw artifacts |

## Required Order

1. Complete the required first reads listed in `agent_protocols/skill_router.md`.
2. Classify the task into one or more rows from the foundation map.
3. Prefer the stable script entry point when one exists; otherwise use the module API.
4. Keep generated full outputs, caches, checkpoints, and trial-level tables under `outputs/` or `scratch/` unless the repository explicitly says to commit a concise summary or config.
5. Run the narrow validation first, then `python scripts/repo_doctor.py fast`.
6. If a real experiment run is finished, run:

```bash
python scripts/repo_doctor.py experiment --route <route_config> --run <run_dir> --gate candidate
```

For route-only smoke checks, run:

```bash
python scripts/repo_doctor.py experiment --route <route_config> --gate smoke
```

## Common Workflows

### Evidence First

Use this when the user asks for formal dataset/split evidence or says results
must be reproducible across machines.

```text
dataset index -> dataset manifest -> data QA -> split manifest -> fast gate
```

Do not claim candidate readiness if the dataset/split manifest is still a
placeholder or lacks subject membership plus `trial_rows`.

### Protocol Runner First

Use this when the user asks to make P1/P2/P3 executable or to estimate workload.

```text
route config -> plan_evaluation_protocol.py -> run_evaluation_protocol.py -> runner manifest
```

The runner manifest is a job plan and lock. It is not a prediction table, score
matrix, or candidate audit report.

For the toy route only, the runner can execute one or more supported jobs:

```bash
python scripts/run_evaluation_protocol.py --protocol p1 --route-config configs/routes/models/toy_eegnet.yaml --run-dir outputs/protocol_runs/toy_p1 --seed 42 --n-folds 2 --execute --execute-gate smoke --max-execute-jobs 1
```

### Artifact First

Use this when the user brings model scores, component scores, or predictions.

```text
component score contract -> score route assembly or prediction writer -> metric report -> run manifest -> experiment audit
```

Preserve metadata columns needed for grouping and audit. `subject_id`,
`trial_id`, `crop_id`, `seed`, and `fold` are metadata for grouping/alignment,
not model features.

### Promotion First

Use this when the user asks whether a route can be promoted.

```text
passing candidate audit report -> promotion audit template -> check_promotion_audit.py -> promoted gate
```

Do not write `PROMOTED` into a route config unless the promoted gate supports it.

## Forbidden Shortcuts

- Do not create ad hoc giant scripts in `scripts/` to bypass these foundations.
- Do not make up checksum, split, metric, environment, or artifact hashes.
- Do not treat `protocol_run_manifest.json` as training evidence.
- Do not commit checkpoints, raw outputs, full prediction tables, caches, or temporary dumps.
- Do not use public/private labels, leaderboard feedback, filenames, `subject_id`, `trial_id`, or `pseudo_trial_id` as model features.

## Reporting

All agent summaries for this repository should say which foundation row was
used, which command or API produced the artifact, and which gate passed. If a
required foundation artifact is still missing, say that explicitly and keep the route status unchanged.
