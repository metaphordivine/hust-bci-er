# car_fbstcnet_as_query_srfnet_reference_whitening_eps3e4_conformer_context_fusion Summary

route_id: car_fbstcnet_as_query_srfnet_reference_whitening_eps3e4_conformer_context_fusion
route_status: IDEA
audit_decision: PASS_P1_P2_FIXED_CROPS_AND_P3
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.7363766933333333
decision: KEEP_AS_PR41_SIBLING_NOT_NEW_CHAMPION
reproduce: python scripts/run_score_fusion_routes.py --route-filter car_fbstcnet_as_query_srfnet_reference_whitening_eps3e4_conformer_context_fusion --export-missing --audit
dataset: train_v1
split: component protocol splits
seed: component manifests
protocol: score_fusion query_context
risk notes: Uses genuine component score matrices. P1 and P3 pass candidate audit; the mixed P2 aggregate is diagnostic-only, while per-fixed-crop P2 assemblies pass candidate audit. Do not tune alpha or temperature from P3 outer/final results.

## Route Family

This route extends the PR41 CAR query-context family by keeping the CAR
FBSTCNet query and adding both SRFNet reference and SRFNet whitening eps3e4 as
context components. The intent is to test whether the balanced SRFNet eps3e4
component can improve DEP/HC or P2 worst behavior without replacing the
stronger CAR query.

## Promotion Checks

- Require genuine score matrices and source manifests for every component.
- Lightweight gate: P1 seed42 >= 0.7178, P2 mean >= 0.7037, or P2 worst > 0.5208.
- DEP/HC must be >= 0.80 to continue and should be >= 0.82 before P3 expansion.
- P3 nested selection may only use inner folds for any future alpha or temperature tuning; outer/final folds are report-only.
- Stop if component keys are misaligned across CAR, SRFNet reference, SRFNet whitening eps3e4, and Conformer evidence.

## Remote Evidence

Runs assembled on B from PR46 commit `f6f71ef6a25e86559e3370a8d775e9637d7c524f`:

- P1 seed42: `outputs/pr46_B_20260522_1837_car_query_srf_ref_context_p1/car_fbstcnet_as_query_srfnet_reference_whitening_eps3e4_conformer_context_fusion`.
  - Audit: PASS.
  - exact BA: 0.7316528.
  - DEP mean BA: 0.6546640; HC mean BA: 0.7701472; DEP/HC: 0.8500505.
- P2 diagnostic: `outputs/pr46_B_20260522_1840_car_query_srf_ref_context_p2/car_fbstcnet_as_query_srfnet_reference_whitening_eps3e4_conformer_context_fusion`.
  - Audit: DIAGNOSTIC_ONLY. Candidate audit fails on P2 crop provenance semantics, not synthetic evidence.
  - exact BA: 0.7161225.
  - DEP mean BA: 0.6345352; HC mean BA: 0.7569161; DEP/HC: 0.8383164.
  - Crop means: crop1 0.7292, crop2 0.7083, crop3 0.8125, crop4 0.7708, crop5 0.7292, random 0.7500, worst 0.5000.
- P2 fixed-crop candidate repair: `outputs/remote_fix_pr46_p2_crop_candidate_20260522_233648/p2__eval_crop*/car_fbstcnet_as_query_srfnet_reference_whitening_eps3e4_conformer_context_fusion`.
  - Audit: PASS for crop1 through crop5.
  - crop exact BA: 0.7291667, 0.7083333, 0.8125000, 0.7708333, 0.7291667.
  - crop DEP/HC: 0.9166667, 0.8333333, 0.8888889, 0.9600000, 0.8000000.
- P3 nested: `outputs/pr46_B_20260522_1843_car_query_srf_ref_context_p3/car_fbstcnet_as_query_srfnet_reference_whitening_eps3e4_conformer_context_fusion`.
  - Audit: PASS.
  - exact BA: 0.7363767.
  - DEP mean BA: 0.6506267; HC mean BA: 0.7792517; DEP/HC: 0.8349378.
  - Final outer fold means: 0.7055, 0.7355, 0.7541, 0.6848, 0.8091.

Interpretation: the route passes the lightweight gate and P3 threshold, but it
does not clearly beat the PR41 CAR/SRFNet-whitening/Conformer context route.
Keep it as a sibling ablation and component-balance reference rather than a
new primary champion.
