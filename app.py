from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from src.config import DEFAULT_THRESHOLD, MODEL_DIR
from src.modeling import predict_churn_probability

MODEL_PATH = MODEL_DIR / "best_churn_model.joblib"


# Preset definitions - mỗi preset chứa tất cả giá trị cần thiết để điền form.
# Key trong "values" phải khớp chính xác với key= của từng widget Streamlit.
CUSTOMER_PRESETS = {
    "high_risk": {
        "label": "🔴 Rủi ro cao",
        "description": "Month-to-month, Fiber optic, tenure thấp, Electronic check",
        "values": {
            "gender": "Female", "senior": "No", "partner": "No", "dependents": "No",
            "tenure": 2, "phone_service": "Yes", "multiple_lines": "No",
            "internet_service": "Fiber optic", "online_security": "No",
            "online_backup": "No", "device_protection": "No", "tech_support": "No",
            "streaming_tv": "Yes", "streaming_movies": "Yes",
            "contract": "Month-to-month", "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 90.0, "total_charges": 180.0,
            "threshold": DEFAULT_THRESHOLD,
        },
    },
    "low_risk": {
        "label": "🟢 Rủi ro thấp",
        "description": "Two year, DSL, tenure cao, Credit card (automatic)",
        "values": {
            "gender": "Male", "senior": "No", "partner": "Yes", "dependents": "Yes",
            "tenure": 60, "phone_service": "Yes", "multiple_lines": "Yes",
            "internet_service": "DSL", "online_security": "Yes",
            "online_backup": "Yes", "device_protection": "Yes", "tech_support": "Yes",
            "streaming_tv": "No", "streaming_movies": "No",
            "contract": "Two year", "paperless_billing": "No",
            "payment_method": "Credit card (automatic)",
            "monthly_charges": 55.0, "total_charges": 3300.0,
            "threshold": DEFAULT_THRESHOLD,
        },
    },
    "senior": {
        "label": "👴 Khách cao tuổi",
        "description": "SeniorCitizen, Fiber optic, chi phí cao",
        "values": {
            "gender": "Male", "senior": "Yes", "partner": "No", "dependents": "No",
            "tenure": 10, "phone_service": "Yes", "multiple_lines": "No",
            "internet_service": "Fiber optic", "online_security": "No",
            "online_backup": "Yes", "device_protection": "Yes", "tech_support": "No",
            "streaming_tv": "Yes", "streaming_movies": "No",
            "contract": "Month-to-month", "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 85.0, "total_charges": 850.0,
            "threshold": DEFAULT_THRESHOLD,
        },
    },
    "new_customer": {
        "label": "🆕 Khách mới",
        "description": "Tenure = 1, dịch vụ cơ bản, DSL",
        "values": {
            "gender": "Female", "senior": "No", "partner": "No", "dependents": "No",
            "tenure": 1, "phone_service": "Yes", "multiple_lines": "No",
            "internet_service": "DSL", "online_security": "No",
            "online_backup": "No", "device_protection": "No", "tech_support": "Yes",
            "streaming_tv": "No", "streaming_movies": "No",
            "contract": "Month-to-month", "paperless_billing": "Yes",
            "payment_method": "Bank transfer (automatic)",
            "monthly_charges": 45.0, "total_charges": 45.0,
            "threshold": DEFAULT_THRESHOLD,
        },
    },
}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Chưa tìm thấy mô hình. Hãy chạy `python train.py` trước khi mở ứng dụng.")
        st.stop()
    return joblib.load(MODEL_PATH)


def _apply_preset(preset_key: str) -> None:
    """Callback: điền toàn bộ session_state để mọi widget tự động nhận giá trị preset."""
    preset = CUSTOMER_PRESETS[preset_key]["values"].copy()
    # Đảm bảo các trường phụ thuộc nhất quán với dịch vụ cha
    if preset.get("internet_service") == "No":
        for field in ("online_security", "online_backup", "device_protection",
                      "tech_support", "streaming_tv", "streaming_movies"):
            preset[field] = "No internet service"
    if preset.get("phone_service") == "No":
        preset["multiple_lines"] = "No phone service"
    for key, value in preset.items():
        st.session_state[key] = value


def main() -> None:
    st.set_page_config(page_title="Telco Customer Churn Prediction", page_icon="📉", layout="wide")
    st.title("📉 Customer Churn Prediction System")
    st.write("Nhập thông tin khách hàng để dự đoán khả năng rời bỏ dịch vụ.")

    # Khởi tạo lịch sử dự đoán (tồn tại trong phiên làm việc)
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []

    model = load_model()

    tab1, tab2 = st.tabs(["Single Prediction", "Bulk Prediction (File)"])

    with tab1:
        # Feature 1: Quick-Fill Preset Buttons 
        st.subheader("📋 Quick-Fill Templates")
        preset_cols = st.columns(len(CUSTOMER_PRESETS))
        for col, (key, preset) in zip(preset_cols, CUSTOMER_PRESETS.items()):
            with col:
                st.button(
                    preset["label"],
                    help=preset["description"],
                    on_click=_apply_preset,
                    args=(key,),
                    use_container_width=True,
                )
        st.divider()

        # Form Inputs 
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"], key="gender")
            senior = st.selectbox(
                "Senior Citizen", ["No", "Yes"],
                format_func=lambda x: "Yes" if x == "Yes" else "No",
                key="senior",
            )
            partner = st.selectbox("Partner", ["No", "Yes"], key="partner")
            dependents = st.selectbox("Dependents", ["No", "Yes"], key="dependents")
            tenure = st.slider("Số tháng sử dụng dịch vụ", 0, 72, 12, key="tenure")

            # PhoneService & MultipleLines (phụ thuộc)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"], key="phone_service")
            no_phone = phone_service == "No"
            multiple_lines = st.selectbox(
                "Multiple Lines",
                options=["No phone service"] if no_phone else ["No", "Yes"],
                disabled=no_phone,
                key="multiple_lines",
                help="Not available when there is no phone service." if no_phone else None,
            )
            if no_phone:
                multiple_lines = "No phone service"

            # InternetService & các dịch vụ phụ thuộc (cột 1)
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet_service")
            no_internet = internet_service == "No"
            inet_help = "Not available when there is no Internet service." if no_internet else None

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

            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], key="paperless_billing")
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                key="payment_method",
            )
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, key="monthly_charges")

            # Feature 2: Auto-calculate TotalCharges
            auto_calc = st.checkbox("Auto-calculate from tenure × MonthlyCharges", key="auto_calc_total")
            if auto_calc:
                total_charges = round(tenure * monthly_charges, 2)
                st.info(f"💰 TotalCharges = {tenure} × {monthly_charges:,.2f} = **{total_charges:,.2f}**")
            else:
                total_charges = st.number_input(
                    "Total Charges", min_value=0.0, max_value=10000.0, value=840.0, key="total_charges",
                )

            threshold = st.slider("Prediction Threshold", 0.30, 0.70, DEFAULT_THRESHOLD, 0.05, key="threshold")

        st.divider()
        submitted = st.button("🔍 Predict Churn", type="primary", use_container_width=True)

        if submitted:
            # Xóa kết quả cũ trước khi tính toán lại
            st.session_state.pop("pred_result", None)
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
            st.session_state["pred_result"] = {
                "probability": probability,
                "label": label,
            }

            # Feature 3: Lưu vào lịch sử dự đoán
            risk = "Cao" if probability >= 0.7 else "Trung bình" if probability >= 0.4 else "Thấp"
            st.session_state["prediction_history"].append({
                "Contract": contract,
                "InternetService": internet_service,
                "Tenure": tenure,
                "MonthlyCharges": monthly_charges,
                "Probability": f"{probability:.2%}",
                "Prediction": label,
                "Risk": risk,
            })

        # Hiển thị kết quả dự đoán
        if "pred_result" in st.session_state:
            probability = st.session_state["pred_result"]["probability"]
            label = st.session_state["pred_result"]["label"]
            risk = "Cao" if probability >= 0.7 else "Trung bình" if probability >= 0.4 else "Thấp"
            risk_color = "🔴" if risk == "Cao" else "🟡" if risk == "Trung bình" else "🟢"

            st.subheader("Kết quả dự đoán")
            st.metric("Xác suất Churn", f"{probability:.2%}")
            st.write(f"**Dự đoán:** {'Có khả năng rời bỏ dịch vụ' if label == 'Yes' else 'Ít khả năng rời bỏ dịch vụ'}")
            st.write(f"**Mức rủi ro:** {risk_color} {risk}")

        # Feature 3: Bảng lịch sử dự đoán
        if st.session_state["prediction_history"]:
            st.divider()
            st.subheader("📊 Lịch sử dự đoán")
            history_df = pd.DataFrame(st.session_state["prediction_history"])
            history_df.index = range(1, len(history_df) + 1)
            history_df.index.name = "#"
            st.dataframe(history_df, use_container_width=True)
            if st.button("🗑 Xóa lịch sử"):
                st.session_state["prediction_history"] = []
                st.session_state.pop("pred_result", None)
                st.rerun()

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
