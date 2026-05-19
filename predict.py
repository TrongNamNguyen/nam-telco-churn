from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.config import DEFAULT_THRESHOLD, MODEL_DIR
from src.data_processing import make_single_customer_example
from src.modeling import predict_churn_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dự đoán khả năng rời bỏ dịch vụ của một khách hàng Telco.")
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Chuỗi JSON chứa thông tin khách hàng. Nếu bỏ trống, chương trình dùng ví dụ mẫu.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_DIR / "best_churn_model.joblib",
        help="Đường dẫn tới mô hình đã huấn luyện.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Ngưỡng phân loại Churn. Mặc định là 0.5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError("Chưa có mô hình. Hãy chạy: python train.py")

    customer = json.loads(args.json) if args.json else make_single_customer_example()
    model = joblib.load(args.model_path)
    result = predict_churn_probability(model, customer, threshold=args.threshold)
    probability = float(result.loc[0, "churn_probability"])
    label = result.loc[0, "prediction_label"]
    risk = "Cao" if probability >= 0.7 else "Trung bình" if probability >= 0.4 else "Thấp"

    print(
        json.dumps(
            {
                "prediction_label": label,
                "churn_probability": round(probability, 4),
                "risk_level": risk,
                "threshold": args.threshold,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
