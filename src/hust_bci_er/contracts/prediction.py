"""Prediction-table schema helpers used by experiment audits."""

from __future__ import annotations


SUBJECT_ID_COLUMNS = ("subject_id", "user_id")
TRIAL_ID_COLUMNS = ("trial_id",)
TOP4_COLUMNS = ("pred_top4", "y_pred_top4")
SCORE_COLUMNS = ("y_score", "score", "probability", "logit")
PREDICTION_COLUMNS = ("y_pred", "pred")
TRUTH_COLUMNS = ("y_true",)
METADATA_COLUMNS = set(SUBJECT_ID_COLUMNS + TRIAL_ID_COLUMNS + ("fold", "seed", "crop_id"))
LABEL_ONLY_COLUMNS = set(TRUTH_COLUMNS)
MODEL_FORBIDDEN_COLUMNS = set(SUBJECT_ID_COLUMNS + TRIAL_ID_COLUMNS + ("filename", "file_name"))


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


def canonical_prediction_column(name: str, schema: dict[str, str | None]) -> str | None:
    if name in {"subject_id", "user_id"}:
        return schema.get("subject_id")
    if name == "trial_id":
        return schema.get("trial_id")
    if name in {"pred_top4", "y_pred_top4"}:
        return schema.get("pred_top4")
    if name in {"score", "y_score", "probability", "logit"}:
        return schema.get("score")
    if name in {"y_pred", "pred"}:
        return schema.get("y_pred")
    if name == "y_true":
        return schema.get("y_true")
    return None
