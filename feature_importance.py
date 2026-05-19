from __future__ import annotations

import joblib

from src.config import FIGURE_DIR, MODEL_DIR, REPORT_DIR
from src.modeling import extract_feature_importance
from src.visualization import plot_feature_importance


def main() -> None:
    model_path = MODEL_DIR / "best_churn_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Chưa có mô hình. Hãy chạy: python train.py")

    model = joblib.load(model_path)
    importance_df = extract_feature_importance(model, top_n=15)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "feature_importance_top15.csv"
    importance_df.to_csv(output_path, index=False)
    plot_feature_importance(importance_df)
    print(importance_df.to_string(index=False))
    print(f"Đã lưu feature importance tại: {output_path}")


if __name__ == "__main__":
    main()
