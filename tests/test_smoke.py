from __future__ import annotations
import pytest
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, RAW_DATA_PATH, TEST_SIZE
from src.data_processing import clean_telco_data, load_telco_data, make_single_customer_example, split_features_target
from src.modeling import build_candidate_models, predict_churn_probability

@pytest.mark.smoke
def test_load_and_split_data():
    df = clean_telco_data(load_telco_data(RAW_DATA_PATH))
    X, y = split_features_target(df)
    assert X.shape[0] == y.shape[0]
    assert "TotalCharges" in X.columns
    assert set(y.unique()) <= {0, 1}

@pytest.mark.smoke
def test_train_one_model_and_predict_probability():
    df = clean_telco_data(load_telco_data(RAW_DATA_PATH))
    X, y = split_features_target(df)
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    model = build_candidate_models()["Logistic Regression"]
    model.fit(X_train, y_train)
    result = predict_churn_probability(model, make_single_customer_example())
    probability = float(result.loc[0, "churn_probability"])
    assert 0.0 <= probability <= 1.0
    assert result.loc[0, "prediction_label"] in {"Yes", "No"}
