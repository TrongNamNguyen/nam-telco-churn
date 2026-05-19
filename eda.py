from __future__ import annotations

from src.config import FIGURE_DIR, RAW_DATA_PATH, REPORT_DIR
from src.data_processing import clean_telco_data, load_telco_data
from src.visualization import plot_churn_by_category, plot_numeric_by_churn, plot_target_distribution


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = clean_telco_data(load_telco_data(RAW_DATA_PATH))
    plot_target_distribution(df)
    for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport"]:
        plot_churn_by_category(df, col)
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        plot_numeric_by_churn(df, col)

    overview = df.describe(include="all").transpose()
    overview.to_csv(REPORT_DIR / "data_overview.csv")
    missing = df.isna().sum().rename("missing_count").to_frame()
    missing["missing_rate"] = missing["missing_count"] / len(df)
    missing.to_csv(REPORT_DIR / "missing_values.csv")
    print(f"Đã tạo EDA tại thư mục {FIGURE_DIR}")


if __name__ == "__main__":
    main()
