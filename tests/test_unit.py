import pytest
import pandas as pd
import numpy as np
from src.data_processing import _strip_text_columns, _fix_total_charges, validate_schema, SENIOR_CITIZEN_MAP
from src.modeling import predict_binary_from_probability

def test_strip_text_columns():
    df = pd.DataFrame({"col1": ["  A  ", "B  "], "col2": [1, 2]})
    result = _strip_text_columns(df)
    assert result["col1"].tolist() == ["A", "B"]
    assert result["col2"].tolist() == [1, 2]

def test_fix_total_charges_logic():
    df = pd.DataFrame({
        "tenure": [0, 2, 5],
        "MonthlyCharges": [50.0, 100.0, 20.0],
        "TotalCharges": [" ", "200.0", np.nan]
    })
    result = _fix_total_charges(df)
    # 0 * 50 = 0
    assert result.loc[0, "TotalCharges"] == 0.0
    # 200.0 giữ nguyên
    assert result.loc[1, "TotalCharges"] == 200.0
    # 5 * 20 = 100
    assert result.loc[2, "TotalCharges"] == 100.0

def test_validate_schema_missing_col():
    df = pd.DataFrame({"wrong_col": [1]})
    with pytest.raises(ValueError, match="Dữ liệu thiếu các cột bắt buộc"):
        validate_schema(df, require_target=True)

def test_predict_binary_threshold():
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    # Threshold 0.5
    res = predict_binary_from_probability(probs, threshold=0.5)
    assert np.array_equal(res, [0, 0, 1, 1])
    # Threshold 0.3
    res_low = predict_binary_from_probability(probs, threshold=0.3)
    assert np.array_equal(res_low, [0, 1, 1, 1])
