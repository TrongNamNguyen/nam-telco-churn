from __future__ import annotations

import joblib
import pandas as pd
import numpy as np
import streamlit as st

from src.config import DEFAULT_THRESHOLD, MODEL_DIR
from src.modeling import predict_churn_probability

MODEL_PATH = MODEL_DIR / "best_churn_model.joblib"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Chưa tìm thấy mô hình. Hãy chạy `python train.py` trước khi mở ứng dụng.")
        st.stop()
    return joblib.load(MODEL_PATH)


def main() -> None:
    st.set_page_config(page_title="Telco Customer Churn Prediction", page_icon="📉", layout="centered")
    st.title("📉 Hệ thống dự đoán Customer Churn")
    st.write("Nhập thông tin khách hàng để dự đoán khả năng rời bỏ dịch vụ.")

    model = load_model()

    tab1, tab2 = st.tabs(["Dự đoán đơn lẻ", "Dự đoán theo lô (File)"])

    with tab1:
        with st.form("customer_form"):
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Giới tính", ["Female", "Male"])
                senior = st.selectbox("Khách hàng cao tuổi", ["No", "Yes"], format_func=lambda x: "Có" if x == "Yes" else "Không")
                partner = st.selectbox("Có vợ/chồng/đối tác", ["No", "Yes"])
                dependents = st.selectbox("Có người phụ thuộc", ["No", "Yes"])
                tenure = st.slider("Số tháng sử dụng dịch vụ", 0, 72, 12)
                phone_service = st.selectbox("Dịch vụ điện thoại", ["No", "Yes"])
                multiple_lines = st.selectbox("Nhiều đường dây", ["No phone service", "No", "Yes"])
                internet_service = st.selectbox("Dịch vụ Internet", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            with col2:
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                contract = st.selectbox("Loại hợp đồng", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Hóa đơn điện tử", ["No", "Yes"])
                payment_method = st.selectbox(
                    "Phương thức thanh toán",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                )
                monthly_charges = st.number_input("Chi phí hàng tháng", min_value=0.0, max_value=200.0, value=70.0)
                total_charges = st.number_input("Tổng chi tiêu", min_value=0.0, max_value=10000.0, value=840.0)
                threshold = st.slider("Ngưỡng dự đoán", 0.30, 0.70, DEFAULT_THRESHOLD, 0.05)

            submitted = st.form_submit_button("Dự đoán Churn")

        if submitted:
            # Xử lý logic nghiệp vụ để tránh mâu thuẫn dữ liệu
            if phone_service == "No":
                multiple_lines = "No phone service"
            
            if internet_service == "No":
                online_security = "No internet service"
                online_backup = "No internet service"
                device_protection = "No internet service"
                tech_support = "No internet service"
                streaming_tv = "No internet service"
                streaming_movies = "No internet service"

            customer = pd.DataFrame(
                [
                    {
                        "gender": gender,
                        "SeniorCitizen": senior,
                        "Partner": partner,
                        "Dependents": dependents,
                        "tenure": tenure,
                        "PhoneService": phone_service,
                        "MultipleLines": multiple_lines,
                        "InternetService": internet_service,
                        "OnlineSecurity": online_security,
                        "OnlineBackup": online_backup,
                        "DeviceProtection": device_protection,
                        "TechSupport": tech_support,
                        "StreamingTV": streaming_tv,
                        "StreamingMovies": streaming_movies,
                        "Contract": contract,
                        "PaperlessBilling": paperless_billing,
                        "PaymentMethod": payment_method,
                        "MonthlyCharges": monthly_charges,
                        "TotalCharges": total_charges,
                    }
                ]
            )
            result = predict_churn_probability(model, customer, threshold=threshold)
            probability = float(result.loc[0, "churn_probability"])
            label = result.loc[0, "prediction_label"]
            risk = "Cao" if probability >= 0.7 else "Trung bình" if probability >= 0.4 else "Thấp"

            st.subheader("Kết quả dự đoán")
            st.metric("Xác suất Churn", f"{probability:.2%}")
            st.write(f"**Dự đoán:** {'Có khả năng rời bỏ dịch vụ' if label == 'Yes' else 'Ít khả năng rời bỏ dịch vụ'}")
            st.write(f"**Mức rủi ro:** {risk}")

    with tab2:
        st.subheader("Dự đoán cho danh sách khách hàng")
        st.info(
            "Tải lên file CSV chứa thông tin khách hàng. File cần có các cột tương tự tập dữ liệu gốc "
            "(ngoại trừ cột Churn). Các giá trị rỗng sẽ được hệ thống tự động xử lý theo logic ETL."
        )
        uploaded_file = st.file_uploader("Tải file CSV khách hàng", type=["csv"])
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            try:
                with st.spinner("Đang xử lý..."):
                    # Reset index để tránh lỗi lệch index khi concat với kết quả dự đoán (vốn có index 0, 1, 2...)
                    input_df = input_df.reset_index(drop=True)
                    results = predict_churn_probability(model, input_df, threshold=DEFAULT_THRESHOLD)
                    final_df = pd.concat([input_df, results], axis=1)
                    st.write(f"Đã dự đoán cho **{len(final_df)}** khách hàng. Hiển thị 20 dòng đầu:")
                    st.dataframe(final_df.head(20))
                    
                    csv = final_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Tải kết quả dự đoán (.csv)", csv, "churn_predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Lỗi định dạng file: {e}")

        with st.expander("Cách hiểu kết quả"):
            st.write(
                "Xác suất Churn là xác suất mô hình ước lượng khách hàng có thể rời bỏ dịch vụ. "
                "Ngưỡng dự đoán quyết định khi nào hệ thống gán nhãn Churn = Yes. "
                "Ngưỡng thấp giúp phát hiện nhiều khách hàng rủi ro hơn nhưng có thể tăng cảnh báo sai."
            )


if __name__ == "__main__":
    main()
