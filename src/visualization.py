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
    plt.title("Phân bổ biến mục tiêu Churn")
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
    plt.figure(figsize=(10, 5))
    bars_no = plt.bar(x - width / 2, rate_table["No"], width, label="No")
    bars_yes = plt.bar(x + width / 2, rate_table["Yes"], width, label="Yes")
    
    # Thêm nhãn % lên đầu các cột
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    add_labels(bars_no)
    add_labels(bars_yes)

    plt.title(f"Tỷ lệ Churn theo {column}")
    plt.ylabel("Tỷ lệ (%)")
    plt.xlabel(column)
    plt.ylim(0, 115) # Tăng giới hạn y để đủ chỗ cho nhãn
    plt.xticks(x, rate_table.index.astype(str), rotation=20, ha="right")
    plt.legend(title="Churn")
    return _save_current_figure(path)


def plot_numeric_by_churn(df: pd.DataFrame, column: str, output_dir: Path = FIGURE_DIR) -> Path:
    path = output_dir / f"{column}_by_churn.png"
    groups = [df.loc[df["Churn"] == label, column].dropna() for label in ["No", "Yes"]]
    plt.figure(figsize=(7, 4.5))
    plt.boxplot(groups, labels=["No", "Yes"], showmeans=True)
    plt.title(f"Phân bổ {column} theo Churn")
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
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap='Blues')
    plt.title(f"Ma trận nhầm lẫn (Confusion Matrix) - {model_name}")
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.xticks([0, 1], ["No", "Yes"])
    plt.yticks([0, 1], ["No", "Yes"])
    
    # Gán nhãn thuật ngữ chuyên môn TN, FP, FN, TP
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            label = labels[i][j]
            plt.text(j, i, f"{val}\n({label})", ha="center", va="center", 
                     color="white" if val > matrix.max()/2 else "black", fontweight='bold')
            
    plt.colorbar(fraction=0.046, pad=0.04)
    return _save_current_figure(path)


def plot_feature_importance(importance_df: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    path = output_dir / "feature_importance_top15.png"
    data = importance_df.sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(data["feature"], data["importance"])
    plt.title("Các đặc trưng quan trọng nhất của mô hình")
    plt.xlabel("Độ quan trọng (Importance)")
    plt.ylabel("Đặc trưng (Features)")
    return _save_current_figure(path)


def plot_threshold_analysis(threshold_df: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    path = output_dir / "threshold_analysis.png"
    plt.figure(figsize=(8, 4.8))
    for metric in ["precision", "recall", "f1_score"]:
        plt.plot(threshold_df["threshold"], threshold_df[metric], marker="o", label=metric)
    plt.title("Ảnh hưởng của ngưỡng dự đoán đến các chỉ số")
    plt.xlabel("Ngưỡng dự đoán")
    plt.ylabel("Điểm số")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    return _save_current_figure(path)
