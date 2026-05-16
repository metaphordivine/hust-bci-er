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

- `RUN_REPRODUCIBILITY_LOCKED` verifies environment, deterministic seed policy, dataloader worker seed policy, checkpoint selection, crop policy, and lock-file hashes are present.
- `RUN_AUDIT_SCHEMA_VERSION` verifies strict-gate evidence uses the current audit schema; candidate/promoted evidence with `audit_schema_version < 2` is blocked.
- `EVIDENCE_CONFIG_SHA_MATCHES_WORKTREE` verifies the audited route file is the exact config snapshot recorded by the run.
- `EVIDENCE_COMMIT_CONTAINS_ROUTE_AND_IMPLEMENTATION` verifies `manifest.git_commit` contains the route file and the implementation paths required to execute it. New non-`score_fusion` models must have an explicit implementation path mapping; unknown mappings are strict-gate failures.
- `ROUTE_MODEL_KWARGS_PASSTHROUGH` verifies route-level `model.*` kwargs are recorded in manifest provenance and therefore reached the builder path.
- `METRIC_SCORE_MATRIX_SHAPE_COMPATIBLE` verifies the score matrix shape matches the declared metric contract; `exact_single_crop_expected_BA` requires exactly five crop score columns.
- `RUN_SPLIT_EVIDENCE_CONSISTENT` verifies subject lists or fold definitions match `trial_rows` subject membership.
- `PRIMARY_METRIC_RECOMPUTE` recomputes the route's declared `evaluation.primary_metric`.
- `PREDICTION_TOP4_RANKING` verifies `pred_top4` is derived from the score column by the repository Top-4 policy.
- `PREDICTION_TRIAL_ID_UNIQUE`, `PREDICTION_TOP4_BINARY`, and `PREDICTION_TOP4_TRUTH_BALANCE` must pass when Top-4 labels are present.
- `RUN_SPLIT_SUBJECT_DISJOINT` and `RUN_SPLIT_ORIGINAL_TRIAL_DISJOINT` verify run-specific split evidence does not leak subjects or original trials across splits. Candidate split evidence must include subject membership plus trial_rows; placeholder subject-only files are not enough.
- `RUN_DATASET_CHECKSUM_SCHEMA` and `RUN_DATASET_CHECKSUM_COVERAGE` verify dataset checksum evidence uses unique paths, lowercase sha256 values, and covers all declared data sources.
- If split evidence is provided only as `trial_rows`, the audit derives subject membership from those rows, including per-fold rows when a `fold` field is present.
- `RUN_DATASET_CHECKSUM_EXTRA` reports checksum paths that are not declared in `data_sources`; candidate/promoted gates should not carry unresolved dataset checksum warnings.
- Route schema validation must reject invalid `augmentation.search_space` cross products before a run starts. Search candidates must keep `input_window_sec`, window parameters, and metric-required crop counts mutually compatible.

Before marking any route as `CANDIDATE`, the evidence chain must prove:

1. The route config is the exact config used by the run.
2. `manifest.config_sha256` matches the audited route file.
3. `manifest.git_commit` contains the route and implementation code needed to run it.
4. Summary reproduce commands use repository-relative path arguments, not local absolute paths.
5. Route `model.*` kwargs are passed to `build_model` and recorded in manifest provenance.
6. Augmentation crop count matches `score_matrix.csv` columns and evaluation metric semantics.
7. `search_space` is validated by full cross product, not only base values.
8. Dataset manifest crop provenance covers every crop in `score_matrix.csv`.
9. Metric parse or recompute failures are fatal for candidate/promoted gates.
10. New route/model/adapter work includes negative tests for dangerous invalid configurations.

For `promoted` gate, add:

```text
reports/promotion_audits/<route_id>_promotion.md
```

The promotion audit must reference a passing candidate audit report and include the fields shown in `reports/promotion_audits/_template.md`.
The referenced candidate audit report must include passing critical evidence rules such as `MANIFEST_VALID`, `PRIMARY_METRIC_RECOMPUTE`, `PRIMARY_METRIC_REPORTED`, `RUN_DATASET_EVIDENCE_VALID`, and `RUN_SPLIT_EVIDENCE_VALID`; Top-4 routes must also include passing Top-4 semantic rules.

## Route Summary Evidence

For advanced route statuses (`CANDIDATE`, `PROMOTED`, `REJECTED`, `ARCHIVED`), the route summary must bind to audit evidence:

```text
audit_report_path:
manifest_path:
manifest_sha256:
primary_metric_value:
```

`check_summary_consistency.py` validates that these paths stay inside the repository, the manifest hash matches, the manifest is valid JSON, `route_id` and `primary_metric` match the summary, and `primary_metric_value` matches the manifest `metrics` entry.

## Gate Types

| Gate | When | Contents |
|---|---|---|
| Fast gate | every PR | schema, naming, no-leakage tests, metric unit tests |
| Experiment gate | after a local experiment | manifest, prediction shape, metric recompute, summary check |
| Heavy gate | manual or scheduled | full CV, pseudo-public holdout, worst crop stress test |
