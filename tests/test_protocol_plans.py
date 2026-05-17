from hust_bci_er.evaluation.protocols import (
    build_protocol1_plan,
    build_protocol2_plan,
    build_protocol3_plan,
)


def test_protocol1_plan_counts_repeated_group_jobs():
    plan = build_protocol1_plan(["ea_deformer"], seeds=[1, 2], n_folds=3)
    data = plan.as_dict()
    assert data["protocol"] == "p1_repeated_group_kfold"
    assert data["planned_train_jobs"] == 6
    assert data["details"]["train_from_scratch_per_fold"] is True
    assert data["guardrails"]["hidden_test_labels_used"] is False
    assert data["guardrails"]["external_score_feedback_used"] is False


def test_protocol2_plan_counts_crop_policy_evaluations():
    plan = build_protocol2_plan(["ea_deformer"], crop_policies=["crop1", "worst"])
    data = plan.as_dict()
    assert data["protocol"] == "p2_pseudo_public_holdout"
    assert data["planned_train_jobs"] == 1
    assert data["planned_evaluation_jobs"] == 2
    assert data["details"]["single_crop_policy"] is True


def test_protocol3_plan_counts_inner_and_outer_jobs():
    plan = build_protocol3_plan(["ea_deformer"], outer_folds=2, inner_folds=3, grid_sizes={"ea_deformer": 4})
    data = plan.as_dict()
    assert data["protocol"] == "p3_nested_selection"
    assert data["planned_train_jobs"] == 26
    assert data["details"]["outer_test_used_for_selection"] is False
    assert data["details"]["final_retrain_on_outer_train"] is True
    assert data["details"]["inner_checkpoint_reused_for_outer_test"] is False
