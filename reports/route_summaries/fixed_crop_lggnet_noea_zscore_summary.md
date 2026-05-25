# fixed_crop_lggnet_noea_zscore Summary

route_id: fixed_crop_lggnet_noea_zscore
route_status: IDEA
audit_decision: PENDING_FORMAL_RERUN
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: TBD
decision: HOLD
evidence_status: exploratory
reproduce: python scripts/run_evaluation_protocol.py --protocol p1 --route-config configs/routes/models/fixed_crop_lggnet_noea_zscore.yaml --execute --execute-gate candidate

## Purpose

EA ablation for `fixed_crop_ea_lggnet`. The route keeps split-first fixed crops, LGGNet model settings, optimizer, and top-4 inference unchanged while removing `euclidean_alignment` from preprocessing.

## Current Evidence

- B machine P2 crop-stress exploratory run started on 2026-05-25.
- Interim P2 fixed-crop mean: 0.625000; worst fixed crop: 0.583333.

## Risk Notes

- The first remote run used a route config copied to the machine before this summary/config existed in the committed worktree, so candidate audit is expected to block on commit provenance.
- Use as an EA/no-EA control, not as a promotion route, until full P1/P2/P3 evidence is rerun from a commit containing this config and summary.
