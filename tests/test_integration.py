import pytest
import pandas as pd
from src.config import FEATURE_COLUMNS
from src.data_processing import clean_telco_data, split_features_target
from src.modeling import build_candidate_models

@pytest.mark.integration
def test_full_cleaning_and_splitting_flow():
    # Giả lập dữ liệu thô cực kỳ "bẩn"
    raw_data = pd.DataFrame({
        "customerID": ["1", "2", "1"], # Trùng lặp ID
        "gender": [" Male ", "Female", " Male "],
        "SeniorCitizen": [0, "1", 0],
        "Partner": ["No", "Yes", "No"],
        "Dependents": ["No", "No", "No"],
        "tenure": [1, 24, 1],
        "PhoneService": ["Yes", "Yes", "Yes"],
        "MultipleLines": ["No", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "Yes", "No"],
        "OnlineBackup": ["No", "No", "No"],
        "DeviceProtection": ["No", "No", "No"],
        "TechSupport": ["No", "No", "No"],
        "StreamingTV": ["No", "No", "No"],
        "StreamingMovies": ["No", "No", "No"],
        "Contract": ["Month-to-month", "One year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check"],
        "MonthlyCharges": [29.85, 70.7, 29.85],
        "TotalCharges": ["29.85", "1697.1", "29.85"],
        "Churn": ["No", "Yes", "No"]
    })
    
    # 1. Test Clean (phải loại bỏ trùng lặp)
    cleaned_df = clean_telco_data(raw_data)
    assert len(cleaned_df) == 2 
    assert cleaned_df["gender"].iloc[0] == "Male" # Đã strip
    
    # 2. Test Split
    X, y = split_features_target(cleaned_df)
    assert X.shape == (2, len(FEATURE_COLUMNS))
    assert y.tolist() == [0, 1]

@pytest.mark.integration
def test_pipeline_construction():
    models = build_candidate_models()
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    # Kiểm tra xem có Preprocessor không
    lr_pipeline = models["Logistic Regression"]
    assert lr_pipeline.named_steps["preprocess"] is not None
