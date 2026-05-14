# Experiment Audit Protocol

This protocol is platform-neutral. It can be followed by a human, local AI assistant, CI job, or any automation runner.

The protocol does not make scientific decisions by natural language judgment. It calls repository commands and uses exit codes plus audit artifacts.

## Trigger

Run this when an experiment is said to be finished, when a route is proposed for review, or before merging a PR that changes configs, evaluation, audit code, or route reports.

## Required Inputs

- Route config path.
- Run output directory, if an experiment was executed.
- Expected evaluation protocol.
- Intended route status: one of the statuses declared in `configs/statuses.yaml`.

`route_status` is the lifecycle state stored in a route config. `audit_decision` is the result of one audit run and is declared in `configs/audit_decisions.yaml`. `BLOCKED` is an audit decision, not a route status.

## Required Commands

```bash
python scripts/repo_doctor.py fast
```

If a run directory exists:

```bash
python scripts/repo_doctor.py experiment --route <route_config> --run <run_dir> --gate candidate
```

For `smoke` or `diagnostic` route-only checks, `--run` may be omitted. For `candidate` or `promoted`, missing run artifacts are blocking.

The experiment gate writes:

```text
<run_dir>/audit_report.json
<run_dir>/audit_report.md
```

Each failed check must include a rule id, severity, reason, and suggested fix.

Gate behavior:

| Gate | Use | WARN handling |
|---|---|---|
| `smoke` | code-path or tiny-run check | may exit zero with WARN |
| `diagnostic` | analysis-only run | may exit zero with non-critical WARN |
| `candidate` | route comparison evidence | unresolved WARN blocks the gate |
| `promoted` | strongest review evidence | unresolved WARN blocks the gate |

## Decision Rules

| Result | Decision |
|---|---|
| leakage, split, or schema guard fails | `REJECT` |
| metrics cannot be recomputed | `DIAGNOSTIC_ONLY` |
| manifest is missing | `DIAGNOSTIC_ONLY` |
| route summary is missing | `BLOCKED` |
| only smoke was run | keep route status as `SMOKE_ONLY`; audit decision is `WARN` unless a stricter check fails |
| all required checks pass | route may advance to its requested lifecycle status |

`WARN` is allowed to keep diagnostic evidence, but it never supports promotion to `CANDIDATE` or `PROMOTED`.

Prediction manifests may set:

```json
{
  "prediction_record_level": "trial",
  "top4_group_keys": ["seed", "fold", "subject_id"]
}
```

Use `top4_group_keys` when a prediction table combines repeated folds or seeds.

Candidate and promoted gates must prove semantic correctness:

- `PRIMARY_METRIC_RECOMPUTE` recomputes the route's declared `evaluation.primary_metric`.
- `PREDICTION_TOP4_RANKING` verifies `pred_top4` is derived from the score column by the repository Top-4 policy.
- `PREDICTION_TRIAL_ID_UNIQUE`, `PREDICTION_TOP4_BINARY`, and `PREDICTION_TOP4_TRUTH_BALANCE` must pass when Top-4 labels are present.
- `RUN_SPLIT_SUBJECT_DISJOINT` and `RUN_SPLIT_ORIGINAL_TRIAL_DISJOINT` verify run-specific split evidence does not leak subjects or original trials across splits.
- `RUN_DATASET_CHECKSUM_SCHEMA` and `RUN_DATASET_CHECKSUM_COVERAGE` verify dataset checksum evidence uses unique paths, lowercase sha256 values, and covers all declared data sources.

For `promoted` gate, add:

```text
reports/promotion_audits/<route_id>_promotion.md
```

The promotion audit must reference a passing candidate audit report and include the fields shown in `reports/promotion_audits/_template.md`.

## Gate Types

| Gate | When | Contents |
|---|---|---|
| Fast gate | every PR | schema, naming, no-leakage tests, metric unit tests |
| Experiment gate | after a local experiment | manifest, prediction shape, metric recompute, summary check |
| Heavy gate | manual or scheduled | full CV, pseudo-public holdout, worst crop stress test |
