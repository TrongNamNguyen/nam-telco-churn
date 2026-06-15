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
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Giới tính", ["Female", "Male"], key="gender")
            senior = st.selectbox(
                "Khách hàng cao tuổi", ["No", "Yes"],
                format_func=lambda x: "Có" if x == "Yes" else "Không",
                key="senior",
            )
            partner = st.selectbox("Có vợ/chồng/đối tác", ["No", "Yes"], key="partner")
            dependents = st.selectbox("Có người phụ thuộc", ["No", "Yes"], key="dependents")
            tenure = st.slider("Số tháng sử dụng dịch vụ", 0, 72, 12, key="tenure")

            # --- PhoneService & MultipleLines (phụ thuộc) ---
            phone_service = st.selectbox("Dịch vụ điện thoại", ["No", "Yes"], key="phone_service")
            no_phone = phone_service == "No"
            multiple_lines = st.selectbox(
                "Nhiều đường dây",
                options=["No phone service"] if no_phone else ["No", "Yes"],
                disabled=no_phone,
                key="multiple_lines",
                help="Không khả dụng khi không có dịch vụ điện thoại." if no_phone else None,
            )
            if no_phone:
                multiple_lines = "No phone service"

            # --- InternetService & các dịch vụ phụ thuộc (cột 1) ---
            internet_service = st.selectbox("Dịch vụ Internet", ["DSL", "Fiber optic", "No"], key="internet_service")
            no_internet = internet_service == "No"
            inet_help = "Không khả dụng khi không có dịch vụ Internet." if no_internet else None

            online_security = st.selectbox(
                "Online Security",
                options=["No internet service"] if no_internet else ["No", "Yes"],
                disabled=no_internet,
                key="online_security",
                help=inet_help,
            )
            if no_internet:
                online_security = "No internet service"

            online_backup = st.selectbox(
                "Online Backup",
                options=["No internet service"] if no_internet else ["No", "Yes"],
                disabled=no_internet,
                key="online_backup",
                help=inet_help,
            )
            if no_internet:
                online_backup = "No internet service"

        with col2:
            device_protection = st.selectbox(
                "Device Protection",
                options=["No internet service"] if no_internet else ["No", "Yes"],
                disabled=no_internet,
                key="device_protection",
                help=inet_help,
            )
            if no_internet:
                device_protection = "No internet service"

            tech_support = st.selectbox(
                "Tech Support",
                options=["No internet service"] if no_internet else ["No", "Yes"],
                disabled=no_internet,
                key="tech_support",
                help=inet_help,
            )
            if no_internet:
                tech_support = "No internet service"

            streaming_tv = st.selectbox(
                "Streaming TV",
                options=["No internet service"] if no_internet else ["No", "Yes"],
                disabled=no_internet,
                key="streaming_tv",
                help=inet_help,
            )
            if no_internet:
                streaming_tv = "No internet service"

            streaming_movies = st.selectbox(
                "Streaming Movies",
                options=["No internet service"] if no_internet else ["No", "Yes"],
                disabled=no_internet,
                key="streaming_movies",
                help=inet_help,
            )
            if no_internet:
                streaming_movies = "No internet service"

            contract = st.selectbox("Loại hợp đồng", ["Month-to-month", "One year", "Two year"], key="contract")
            paperless_billing = st.selectbox("Hóa đơn điện tử", ["No", "Yes"], key="paperless_billing")
            payment_method = st.selectbox(
                "Phương thức thanh toán",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                key="payment_method",
            )
            monthly_charges = st.number_input("Chi phí hàng tháng", min_value=0.0, max_value=200.0, value=70.0, key="monthly_charges")
            total_charges = st.number_input("Tổng chi tiêu", min_value=0.0, max_value=10000.0, value=840.0, key="total_charges")
            threshold = st.slider("Ngưỡng dự đoán", 0.30, 0.70, DEFAULT_THRESHOLD, 0.05, key="threshold")

        st.divider()
        submitted = st.button("🔍 Dự đoán Churn", type="primary", use_container_width=True)

        if submitted:
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
            st.session_state["pred_result"] = {
                "probability": float(result.loc[0, "churn_probability"]),
                "label": result.loc[0, "prediction_label"],
            }

        if "pred_result" in st.session_state:
            probability = st.session_state["pred_result"]["probability"]
            label = st.session_state["pred_result"]["label"]
            risk = "Cao" if probability >= 0.7 else "Trung bình" if probability >= 0.4 else "Thấp"
            risk_color = "🔴" if risk == "Cao" else "🟡" if risk == "Trung bình" else "🟢"

            st.subheader("Kết quả dự đoán")
            st.metric("Xác suất Churn", f"{probability:.2%}")
            st.write(f"**Dự đoán:** {'Có khả năng rời bỏ dịch vụ' if label == 'Yes' else 'Ít khả năng rời bỏ dịch vụ'}")
            st.write(f"**Mức rủi ro:** {risk_color} {risk}")

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
