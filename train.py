from __future__ import annotations

import json

import joblib
from sklearn.model_selection import train_test_split

from src.config import FIGURE_DIR, MODEL_DIR, RANDOM_STATE, RAW_DATA_PATH, REPORT_DIR, TEST_SIZE
from src.data_processing import clean_telco_data, load_telco_data, split_features_target
from src.modeling import evaluate_thresholds, extract_feature_importance, train_and_evaluate_models
from src.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_metrics,
    plot_threshold_analysis,
)


def main() -> None:
    """
    QUY TRÌNH CHÍNH (Workflow):
    1. Tạo thư mục chứa kết quả (models/, reports/).
    2. Nạp dữ liệu thô (load_telco_data).
    3. Làm sạch dữ liệu (clean_telco_data) - xử lý rỗng, sai định dạng.
    4. Chia X (đầu vào) và y (đầu ra).
    5. Chia tập Train (để AI học) và tập Test (để AI thi thử), tỉ lệ 80/20.
    6. Huấn luyện đồng thời 3 mô hình và so sánh.
    7. Lưu mô hình tốt nhất (F1-score cao nhất) vào ổ cứng để dùng cho Web.
    8. Xuất các báo cáo và biểu đồ (Ma trận nhầm lẫn, Feature Importance).
    """
    # Tạo thư mục nếu chưa có
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Nạp và làm sạch
    raw_df = load_telco_data(RAW_DATA_PATH)
    df = clean_telco_data(raw_df)
    
    # Chia đặc trưng và mục tiêu
    X, y = split_features_target(df)
    
    # Chia Train/Test (Stratify giúp tỉ lệ Churn ở 2 tập giống nhau)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Huấn luyện
    trained_models, metrics_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Chọn và lưu mô hình tốt nhất
    best_model_name = metrics_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, MODEL_DIR / "best_churn_model.joblib")
    
    # Xuất báo cáo CSV
    metrics_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    
    # Vẽ biểu đồ
    plot_model_metrics(metrics_df)
    for model_name, model in trained_models.items():
        plot_confusion_matrix(model, X_test, y_test, model_name)

    # Phân tích ngưỡng cho mô hình tốt nhất
    threshold_df = evaluate_thresholds(best_model, X_test, y_test)
    threshold_df.to_csv(REPORT_DIR / "threshold_analysis.csv", index=False)
    plot_threshold_analysis(threshold_df)

    # Nếu mô hình tốt nhất là Random Forest, ta xem yếu tố nào quan trọng nhất
    feature_importance_path = None
    if best_model_name == "Random Forest":
        importance_df = extract_feature_importance(best_model, top_n=15)
        importance_df.to_csv(REPORT_DIR / "feature_importance_top15.csv", index=False)
        plot_feature_importance(importance_df)
        feature_importance_path = str(REPORT_DIR / "feature_importance_top15.csv")

    # Lưu tóm tắt vào file JSON
    summary = {
        "dataset_shape_after_cleaning": list(df.shape),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "best_model": best_model_name,
        "note": "Mô hình này đã sẵn sàng để triển khai lên Web."
    }
    (REPORT_DIR / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Huấn luyện hoàn tất.")
    print(metrics_df.to_string(index=False))
    print(f"Mô hình tốt nhất: {best_model_name}")
    print(f"Đã lưu mô hình tại: {MODEL_DIR / 'best_churn_model.joblib'}")


if __name__ == "__main__":
    # Đảm bảo Windows hỗ trợ hiển thị tiếng Việt
    import sys
    import io
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()
