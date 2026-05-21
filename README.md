# 📉 Telco Customer Churn Prediction

Dự án dự đoán khả năng rời bỏ mạng của khách hàng viễn thông, sử dụng Machine Learning (Scikit-learn) và triển khai Web App với Streamlit.

## 🚀 Tính năng chính
- **ETL & Data Cleaning:** Tự động xử lý dữ liệu thiếu, đồng nhất định dạng.
- **Model Training:** Huấn luyện đồng thời Logistic Regression, Decision Tree, Random Forest.
- **Evaluation:** Đánh giá chi tiết qua Precision, Recall, F1-score và Confusion Matrix.
- **Web Interface:** Giao diện dự đoán trực quan cho người dùng cuối.
- **Automated Testing:** Hệ thống kiểm thử 3 tầng (Unit, Integration, System).

## 📂 Cấu trúc dự án
- `src/`: Mã nguồn cốt lõi (xử lý dữ liệu, mô hình hóa, trực quan hóa).
- `data/`: Chứa dữ liệu thô (CSV).
- `tests/`: Bộ kiểm thử tự động.
- `train.py`: Quy trình huấn luyện toàn diện.
- `app.py`: Ứng dụng Web Streamlit.

## 🛠 Cài đặt & Sử dụng

1. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Huấn luyện mô hình:**
   ```bash
   python train.py
   ```

3. **Chạy Web App:**
   ```bash
   streamlit run app.py
   ```

4. **Chạy Kiểm thử:**
   ```bash
   $env:PYTHONPATH = "."; pytest
   ```

## 📊 Kết quả
Mô hình **Random Forest** đạt hiệu quả cao nhất với các chỉ số tối ưu cho việc nhận diện khách hàng rủi ro. Cụ thể:
- **F1-score:** 0.6295
- **Recall:** 0.7701
- **Precision:** 0.5323

Các chỉ số này cho thấy mô hình cân bằng tốt giữa việc phát hiện nhiều khách hàng có rủi ro rời mạng (Recall cao) và độ chính xác của dự đoán (Precision hợp lý). Các báo cáo chi tiết và biểu đồ phân tích nằm trong thư mục `reports/`.
