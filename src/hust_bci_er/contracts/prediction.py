"""Prediction-table schema helpers used by experiment audits."""

from __future__ import annotations


SUBJECT_ID_COLUMNS = ("subject_id", "user_id")
TRIAL_ID_COLUMNS = ("trial_id",)
TOP4_COLUMNS = ("pred_top4", "y_pred_top4")
SCORE_COLUMNS = ("y_score", "score", "probability", "logit")
PREDICTION_COLUMNS = ("y_pred", "pred")
TRUTH_COLUMNS = ("y_true",)


def first_present(fields: set[str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in fields:
            return name
    return None


def prediction_schema(fields: set[str]) -> dict[str, str | None]:
    normalized = {str(field) for field in fields}
    return {
        "subject_id": first_present(normalized, SUBJECT_ID_COLUMNS),
        "trial_id": first_present(normalized, TRIAL_ID_COLUMNS),
        "pred_top4": first_present(normalized, TOP4_COLUMNS),
        "score": first_present(normalized, SCORE_COLUMNS),
        "y_pred": first_present(normalized, PREDICTION_COLUMNS),
        "y_true": first_present(normalized, TRUTH_COLUMNS),
    }
