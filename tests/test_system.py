import subprocess
import os
import joblib
import pandas as pd
from src.config import MODEL_DIR, RAW_DATA_PATH

def test_system_training_pipeline():
    """Kiểm tra xem chạy train.py có thực sự tạo ra các file output không."""
    # Chạy script train.py
    result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
    
    # Kiểm tra xem có lỗi không (Exit code 0)
    assert result.returncode == 0
    
    # Kiểm tra các file quan trọng có tồn tại không
    assert os.path.exists(MODEL_DIR / "best_churn_model.joblib")
    assert os.path.exists("reports/model_comparison.csv")
    assert os.path.exists("reports/figures/01_churn_distribution.png")

def test_system_prediction_cli():
    """Kiểm tra tool CLI predict.py."""
    # Đảm bảo có model trước
    if not os.path.exists(MODEL_DIR / "best_churn_model.joblib"):
        subprocess.run(["python", "train.py"])
        
    # Chạy predict.py với ví dụ mặc định
    result = subprocess.run(["python", "predict.py"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "prediction_label" in result.stdout
    assert "churn_probability" in result.stdout
