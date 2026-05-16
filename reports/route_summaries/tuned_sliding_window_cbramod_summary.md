# tuned_sliding_window_cbramod Summary

route_id: tuned_sliding_window_cbramod
route_status: CANDIDATE
audit_decision: PASS
gate: candidate
primary_metric: exact_single_crop_expected_BA
primary_metric_value: 0.571672
decision: PASS
reproduce: python scripts/repo_doctor.py experiment --route configs/routes/models/tuned_sliding_window_cbramod.yaml --run outputs/model_route_exploration/20260516_candidate/tuned_sliding_window_cbramod --gate candidate
dataset: train_v1
split: p1_seed42_fold0
seed: 42
protocol: p1_repeated_group_kfold
risk notes: candidate audit passed; improves over sliding_window_cbramod 0.506853 by +0.064819; compact evidence is stored under reports.
audit_report_path: reports/route_summaries/evidence/tuned_sliding_window_cbramod_audit_report.json
manifest_path: reports/route_summaries/evidence/tuned_sliding_window_cbramod_manifest.json
manifest_sha256: 4f54788a7bc65b377f8a4b20d8278ad1e5a3158df8392ffad8878353a211e016
