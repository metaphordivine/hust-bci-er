from hust_bci_er.contracts.prediction import prediction_schema


def test_prediction_schema_accepts_subject_aliases():
    schema = prediction_schema({"user_id", "trial_id", "y_score", "pred_top4", "y_true", "y_pred"})
    assert schema["subject_id"] == "user_id"
    assert schema["trial_id"] == "trial_id"
    assert schema["score"] == "y_score"
    assert schema["pred_top4"] == "pred_top4"


def test_prediction_schema_prefers_subject_id():
    schema = prediction_schema({"subject_id", "user_id", "trial_id", "score"})
    assert schema["subject_id"] == "subject_id"
