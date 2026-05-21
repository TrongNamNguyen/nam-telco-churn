# 📉 Hệ thống dự báo rời bỏ dịch vụ (Customer Churn Prediction)
Dự án này tập trung vào việc xây dựng một hệ thống học máy (Machine Learning) toàn diện nhằm dự đoán khả năng rời bỏ mạng của khách hàng viễn thông. Hệ thống được thiết kế theo quy chuẩn công nghiệp, bao gồm các giai đoạn từ xử lý dữ liệu (ETL), huấn luyện mô hình, đánh giá chuyên sâu cho đến triển khai ứng dụng Web phục vụ ra quyết định kinh doanh.

## 📖 Giới thiệu bài toán
Trong ngành viễn thông, việc duy trì khách hàng hiện tại có chi phí thấp hơn nhiều so với việc tìm kiếm khách hàng mới. Dự án này sử dụng dữ liệu lịch sử để nhận diện các dấu hiệu rủi ro, giúp doanh nghiệp chủ động đưa ra các chương trình giữ chân khách hàng hiệu quả.

### Thông tin tập dữ liệu (Dataset)
- **Nguồn:** IBM Telco Customer Churn.
- **Quy mô:** 7,043 mẫu dữ liệu khách hàng.
- **Đặc trưng (Features):** 21 cột thông tin bao gồm:
- Thông tin nhân khẩu học (Giới tính, Người phụ thuộc...).
- Thông tin dịch vụ (Internet, Streaming, Tech Support...).
- Thông tin hợp đồng (Loại hợp đồng, Phương thức thanh toán, Chi phí...).
- **Tỉ lệ nhãn (Target):** ~26.5% khách hàng đã rời mạng (Churn = Yes).
## 🏗 Luồng hệ thống (System Flow)
```mermaid
graph TD
   A[Dữ liệu thô (CSV)] --> B[Làm sạch & Đồng nhất định dạng]
   B --> C[Tiền xử lý & Feature Engineering]
   C --> D[Huấn luyện đồng thời 3 mô hình]
   D --> E[Đánh giá qua Precision/Recall/F1]
   E --> F[Lưu mô hình tốt nhất (.joblib)]
   F --> G[Triển khai Streamlit Web App]
   G --> H[Dự đoán đơn lẻ/theo lô]
```

## 📊 Kết quả thực nghiệm
Sau quá trình huấn luyện và so sánh giữa Logistic Regression, Decision Tree và Random Forest, mô hình **Random Forest** được lựa chọn làm mô hình cốt lõi nhờ khả năng cân bằng tốt giữa độ chính xác và khả năng nhận diện khách hàng rủi ro.

| Chỉ số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| **Accuracy** | 0.7594 | Độ chính xác tổng thể trên tập kiểm thử. |
| **Recall** | **0.7701** | Khả năng phát hiện đúng khách hàng thực sự rời mạng (Quan trọng nhất). |
| **Precision** | 0.5323 | Độ tin cậy của các cảnh báo rời mạng được đưa ra. |
| **F1-score** | 0.6295 | Chỉ số trung bình hài hòa giữa Precision và Recall. |

> **Nhận xét:** Với Recall đạt 77%, hệ thống cho phép doanh nghiệp bao phủ được phần lớn nhóm khách hàng có nguy cơ cao, dù chấp nhận một tỉ lệ nhỏ cảnh báo sai (Precision 53%) để đảm bảo không bỏ sót đối tượng quan trọng.

## 📂 Cấu trúc dự án
- `src/`: Chứa mã nguồn logic chính (Clean, Train, Plot).
- `data/raw/`: Chứa tệp dữ liệu gốc `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- `models/`: Lưu trữ mô hình đã huấn luyện thành công.
- `reports/`: Chứa các báo cáo định lượng và biểu đồ phân tích (Confusion Matrix, Feature Importance).