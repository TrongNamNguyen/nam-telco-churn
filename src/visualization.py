from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURE_DIR
from src.modeling import get_confusion_matrix


def _save_current_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def plot_target_distribution(df: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    path = output_dir / "01_churn_distribution.png"
    counts = df["Churn"].value_counts().reindex(["No", "Yes"])
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Phân bố biến mục tiêu Churn")
    plt.xlabel("Churn")
    plt.ylabel("Số lượng khách hàng")
    for idx, value in enumerate(counts.values):
        plt.text(idx, value, str(int(value)), ha="center", va="bottom")
    return _save_current_figure(path)


def plot_churn_by_category(df: pd.DataFrame, column: str, output_dir: Path = FIGURE_DIR) -> Path:
    path = output_dir / f"churn_by_{column}.png"
    rate_table = (
        pd.crosstab(df[column], df["Churn"], normalize="index")
        .mul(100)
        .reindex(columns=["No", "Yes"])
        .fillna(0)
    )
    x = np.arange(len(rate_table.index))
    width = 0.38
    plt.figure(figsize=(9, 4.8))
    plt.bar(x - width / 2, rate_table["No"], width, label="No")
    plt.bar(x + width / 2, rate_table["Yes"], width, label="Yes")
    plt.title(f"Tỷ lệ Churn theo {column}")
    plt.ylabel("Tỷ lệ (%)")
    plt.xlabel(column)
    plt.xticks(x, rate_table.index.astype(str), rotation=20, ha="right")
    plt.legend(title="Churn")
    return _save_current_figure(path)


def plot_numeric_by_churn(df: pd.DataFrame, column: str, output_dir: Path = FIGURE_DIR) -> Path:
    path = output_dir / f"{column}_by_churn.png"
    groups = [df.loc[df["Churn"] == label, column].dropna() for label in ["No", "Yes"]]
    plt.figure(figsize=(7, 4.5))
    plt.boxplot(groups, labels=["No", "Yes"], showmeans=True)
    plt.title(f"Phân bố {column} theo Churn")
    plt.xlabel("Churn")
    plt.ylabel(column)
    return _save_current_figure(path)


def plot_model_metrics(metrics_df: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    path = output_dir / "model_metric_comparison.png"
    metric_cols = ["accuracy", "precision", "recall", "f1_score"]
    models = metrics_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.18
    plt.figure(figsize=(10, 5))
    for idx, metric in enumerate(metric_cols):
        plt.bar(x + (idx - 1.5) * width, metrics_df[metric], width, label=metric)
    plt.title("So sánh hiệu quả các mô hình")
    plt.ylim(0, 1)
    plt.xlabel("Mô hình")
    plt.ylabel("Điểm số")
    plt.xticks(x, models, rotation=10)
    plt.legend()
    return _save_current_figure(path)


def plot_confusion_matrix(model, X_test, y_test, model_name: str, output_dir: Path = FIGURE_DIR) -> Path:
    path = output_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    matrix = get_confusion_matrix(model, X_test, y_test)
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.xticks([0, 1], ["No", "Yes"])
    plt.yticks([0, 1], ["No", "Yes"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")
    plt.colorbar(fraction=0.046, pad=0.04)
    return _save_current_figure(path)


def plot_feature_importance(importance_df: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    path = output_dir / "feature_importance_top15.png"
    data = importance_df.sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(data["feature"], data["importance"])
    plt.title("Top 15 đặc trưng quan trọng của Random Forest")
    plt.xlabel("Feature importance")
    plt.ylabel("Đặc trưng")
    return _save_current_figure(path)


def plot_threshold_analysis(threshold_df: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    path = output_dir / "threshold_analysis.png"
    plt.figure(figsize=(8, 4.8))
    for metric in ["precision", "recall", "f1_score"]:
        plt.plot(threshold_df["threshold"], threshold_df[metric], marker="o", label=metric)
    plt.title("Ảnh hưởng của ngưỡng dự đoán đến Precision, Recall và F1-score")
    plt.xlabel("Ngưỡng dự đoán")
    plt.ylabel("Điểm số")
    plt.ylim(0, 1)
    plt.legend()
    return _save_current_figure(path)
