import pytest
import subprocess
import sys
from src.config import MODEL_DIR, REPORT_DIR, FIGURE_DIR

@pytest.mark.system
def test_system_training_pipeline():
    """Kiểm tra xem chạy train.py có thực sự tạo ra các file output không."""
    # Chạy script train.py
    result = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True, encoding="utf-8")
    
    # Kiểm tra xem có lỗi không (Exit code 0)
    assert result.returncode == 0
    
    # Kiểm tra các file quan trọng có tồn tại không
    assert (MODEL_DIR / "best_churn_model.joblib").exists()
    assert (REPORT_DIR / "model_comparison.csv").exists()
    assert (FIGURE_DIR / "model_metric_comparison.png").exists()

@pytest.mark.system
def test_system_prediction_cli():
    """Kiểm tra tool CLI predict.py."""
    # Đảm bảo có model trước
    if not (MODEL_DIR / "best_churn_model.joblib").exists():
        subprocess.run([sys.executable, "train.py"], encoding="utf-8")
        
    # Chạy predict.py với ví dụ mặc định
    result = subprocess.run([sys.executable, "predict.py"], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0
    assert "prediction_label" in result.stdout
    assert "churn_probability" in result.stdout
