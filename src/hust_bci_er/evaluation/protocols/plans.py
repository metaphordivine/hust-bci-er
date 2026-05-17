"""Evaluation protocol planning helpers.

These functions define the contract and workload for the three rigorous
evaluation protocols. They do not train models and do not read labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


DEFAULT_SEEDS = (42, 123, 456, 789, 1024)
DEFAULT_CROP_POLICIES = ("crop1", "crop2", "crop3", "crop4", "crop5", "random", "worst")


@dataclass(frozen=True)
class ProtocolPlan:
    protocol: str
    use_case: str
    primary_metric: str
    planned_train_jobs: int
    planned_evaluation_jobs: int
    output_artifacts: tuple[str, ...]
    guardrails: Mapping[str, bool]
    details: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "use_case": self.use_case,
            "primary_metric": self.primary_metric,
            "planned_train_jobs": self.planned_train_jobs,
            "planned_evaluation_jobs": self.planned_evaluation_jobs,
            "output_artifacts": list(self.output_artifacts),
            "guardrails": dict(self.guardrails),
            "details": dict(self.details),
        }


def default_guardrails() -> dict[str, bool]:
    return {
        "hidden_test_labels_used": False,
        "external_score_feedback_used": False,
        "final_submission_generated": False,
        "cv_split_modified": False,
        "id_fields_used_as_features": False,
    }


def route_list(routes: Sequence[str]) -> list[str]:
    clean = [str(route) for route in routes if str(route)]
    if not clean:
        raise ValueError("at least one route_id is required")
    if len(set(clean)) != len(clean):
        raise ValueError("route_id values must be unique")
    return clean


def build_protocol1_plan(
    routes: Sequence[str],
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_folds: int = 5,
    val_fraction: float = 0.2,
) -> ProtocolPlan:
    """Repeated subject-group k-fold retraining.

    Use this for strict route comparison when compute budget allows repeated
    from-scratch retraining.
    """
    route_ids = route_list(routes)
    seed_values = [int(seed) for seed in seeds]
    if n_folds <= 1:
        raise ValueError("n_folds must be greater than 1")
    if not 0 < float(val_fraction) < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    train_jobs = len(route_ids) * len(seed_values) * int(n_folds)
    return ProtocolPlan(
        protocol="p1_repeated_group_kfold",
        use_case="严格重训比较；适合判断路线平均水平和 seed/fold 稳定性。",
        primary_metric="exact_single_crop_expected_BA",
        planned_train_jobs=train_jobs,
        planned_evaluation_jobs=train_jobs,
        output_artifacts=("protocol1_board.csv", "protocol1_subject_ba.csv", "protocol1_audit.json"),
        guardrails=default_guardrails(),
        details={
            "routes": route_ids,
            "seeds": seed_values,
            "n_folds": int(n_folds),
            "val_fraction": float(val_fraction),
            "subject_group_split": True,
            "train_from_scratch_per_fold": True,
            "secondary_metrics": ["exact_assignment_all_correct_rate", "fold_std", "subject_std"],
        },
    )


def build_protocol2_plan(
    routes: Sequence[str],
    *,
    n_holdout_subjects: int = 12,
    holdout_seed: int = 999,
    train_seed: int = 42,
    crop_policies: Sequence[str] = DEFAULT_CROP_POLICIES,
) -> ProtocolPlan:
    """Pseudo-public subject holdout with single-crop policies.

    Use this for crop-position and held-out-subject robustness checks.
    """
    route_ids = route_list(routes)
    policy_values = [str(policy) for policy in crop_policies]
    if n_holdout_subjects <= 0:
        raise ValueError("n_holdout_subjects must be positive")
    if not policy_values:
        raise ValueError("at least one crop policy is required")
    train_jobs = len(route_ids)
    eval_jobs = len(route_ids) * len(policy_values)
    return ProtocolPlan(
        protocol="p2_pseudo_public_holdout",
        use_case="模拟未知 subject 和单 crop 推理；适合检查 crop 敏感性、worst crop 抗性和 no-Top4 对照。",
        primary_metric="exact_single_crop_expected_BA",
        planned_train_jobs=train_jobs,
        planned_evaluation_jobs=eval_jobs,
        output_artifacts=("protocol2_board.csv", "protocol2_subject_ba.csv", "protocol2_audit.json"),
        guardrails=default_guardrails(),
        details={
            "routes": route_ids,
            "n_holdout_subjects": int(n_holdout_subjects),
            "holdout_seed": int(holdout_seed),
            "train_seed": int(train_seed),
            "crop_policies": policy_values,
            "subject_group_split": True,
            "single_crop_policy": True,
            "secondary_metrics": ["crop_policy_mean", "worst_crop_gap", "exact_assignment_all_correct_rate"],
        },
    )


def build_protocol3_plan(
    routes: Sequence[str],
    *,
    outer_folds: int = 5,
    inner_folds: int = 3,
    outer_seed: int = 42,
    inner_seed: int = 123,
    grid_sizes: Mapping[str, int] | None = None,
) -> ProtocolPlan:
    """Nested model selection.

    Use this when route parameters are selected from data. Outer-test subjects
    must never participate in inner selection.
    """
    route_ids = route_list(routes)
    if outer_folds <= 1:
        raise ValueError("outer_folds must be greater than 1")
    if inner_folds <= 1:
        raise ValueError("inner_folds must be greater than 1")
    sizes = {route: int((grid_sizes or {}).get(route, 1)) for route in route_ids}
    if any(size <= 0 for size in sizes.values()):
        raise ValueError("all grid sizes must be positive")
    inner_jobs = sum(sizes.values()) * int(outer_folds) * int(inner_folds)
    outer_jobs = len(route_ids) * int(outer_folds)
    return ProtocolPlan(
        protocol="p3_nested_selection",
        use_case="带参数选择和 outer-train final retrain 的最严格评估；适合最终确认调参路线，不适合日常快速探索。",
        primary_metric="exact_single_crop_expected_BA",
        planned_train_jobs=inner_jobs + outer_jobs,
        planned_evaluation_jobs=inner_jobs + outer_jobs,
        output_artifacts=("protocol3_board.csv", "protocol3_param_selection.csv", "protocol3_subject_ba.csv", "protocol3_audit.json"),
        guardrails=default_guardrails(),
        details={
            "routes": route_ids,
            "outer_folds": int(outer_folds),
            "inner_folds": int(inner_folds),
            "outer_seed": int(outer_seed),
            "inner_seed": int(inner_seed),
            "grid_sizes": sizes,
            "subject_group_split": True,
            "outer_test_used_for_selection": False,
            "final_retrain_on_outer_train": True,
            "inner_checkpoint_reused_for_outer_test": False,
            "secondary_metrics": ["exact_assignment_all_correct_rate", "param_selection_stability", "fold_std"],
        },
    )
