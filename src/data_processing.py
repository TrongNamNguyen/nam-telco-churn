from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CATEGORICAL_FEATURES, EXPECTED_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN

SENIOR_CITIZEN_MAP = {0: "No", 1: "Yes", "0": "No", "1": "Yes", "No": "No", "Yes": "Yes"}


def load_telco_data(path: str | Path) -> pd.DataFrame:
    """
    HÀM NẠP DỮ LIỆU:
    - Nhiệm vụ: Đọc file CSV từ ổ cứng vào bộ nhớ (DataFrame).
    - Giải thích: Kiểm tra xem file có tồn tại không, nếu không sẽ báo lỗi rõ ràng để biết mà kiểm tra đường dẫn.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {path}. Hãy đặt file CSV Telco vào đúng đường dẫn."
        )
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame, *, require_target: bool = True) -> None:
    """
    HÀM KIỂM TRA CẤU TRÚC:
    - Nhiệm vụ: Đảm bảo file dữ liệu nạp vào có đủ các cột cần thiết (như tenure, Contract...).
    - Giải thích: Nếu thiếu cột, AI sẽ không thể 'học' hoặc 'dự đoán' được, nên phải báo lỗi ngay lập tức.
    """
    required_columns = EXPECTED_COLUMNS if require_target else FEATURE_COLUMNS
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {missing}")


def _strip_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Hàm phụ: Xóa khoảng trắng thừa ở đầu/cuối của các cột kiểu chữ."""
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def _fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Hàm phụ: Chuyển TotalCharges về số và điền giá trị thiếu theo logic nghiệp vụ."""
    # 1. Chuyển về số, những ô trống hoặc lỗi sẽ thành NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # 2. Điền NaN bằng MonthlyCharges * tenure (vì thường NaN xuất hiện khi khách mới có tenure=0)
    mask = df["TotalCharges"].isna()
    if mask.any():
        df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"]
    return df


def normalize_customer_input(customer_data: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """
    HÀM CHUẨN HÓA ĐẦU VÀO:
    - Dùng cho Web App hoặc API khi có khách hàng mới cần dự đoán.
    """
    if isinstance(customer_data, dict):
        df = pd.DataFrame([customer_data])
    else:
        df = customer_data.copy()

    # Bước 1: Kiểm tra xem có đủ cột để dự đoán không
    validate_schema(df, require_target=False)

    # Bước 2: Đồng nhất SeniorCitizen (0/1 -> No/Yes)
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map(SENIOR_CITIZEN_MAP)

    # Bước 3: Ép kiểu số cho các cột định lượng
    for col in ["tenure", "MonthlyCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Bước 4: Xử lý TotalCharges bằng logic dùng chung
    df = _fix_total_charges(df)

    # Bước 5: Xóa khoảng trắng ở các cột chữ
    df = _strip_text_columns(df)

    return df[FEATURE_COLUMNS].copy()


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    HÀM LÀM SẠCH DỮ LIỆU TỔNG THỂ:
    - Dùng khi huấn luyện mô hình với tập dữ liệu lớn.
    """
    df = df.copy()
    # Chuẩn hóa tên cột
    df.columns = df.columns.str.strip()
    validate_schema(df)

    # 1. Xử lý cột số và giá trị thiếu
    df = _fix_total_charges(df)

    # 2. Đồng nhất SeniorCitizen
    df["SeniorCitizen"] = df["SeniorCitizen"].map(SENIOR_CITIZEN_MAP)

    # 3. Làm sạch chuỗi và xóa trùng lặp
    df = _strip_text_columns(df)
    df = df.drop_duplicates()
    
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    HÀM CHIA DỮ LIỆU X VÀ Y:
    - X (Features): Các thông tin khách hàng (đầu vào).
    - y (Target): Kết quả Churn 0 (Ở lại) hoặc 1 (Rời bỏ) (đầu ra).
    """
    validate_schema(df)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    if y.isna().any():
        invalid = df.loc[y.isna(), TARGET_COLUMN].unique().tolist()
        raise ValueError(f"Cột Churn có giá trị không hợp lệ: {invalid}")
    return X, y.astype(int)


def make_single_customer_example() -> dict[str, Any]:
    """A realistic high-risk sample input for smoke testing the prediction program."""
    return {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.0,
        "TotalCharges": 180.0,
    }


def make_low_risk_customer_example() -> dict[str, Any]:
    """A realistic low-risk sample input for demo comparison."""
    return {
        "gender": "Male",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 55.0,
        "TotalCharges": 3300.0,
    }
