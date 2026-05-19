from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import CATEGORICAL_FEATURES, DEFAULT_THRESHOLD, NUMERIC_FEATURES, RANDOM_STATE
from src.data_processing import normalize_customer_input


def build_preprocessor() -> ColumnTransformer:
    """
    HÀM XÂY DỰNG BỘ TIỀN XỬ LÝ:
    - Numeric: Điền giá trị rỗng bằng trung vị (median), chuẩn hóa (StandardScaler).
    - Categorical: Điền rỗng bằng giá trị phổ biến nhất, mã hóa 0/1 (OneHot).
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_candidate_models() -> OrderedDict[str, Pipeline]:
    """
    HÀM TẠO CÁC MÔ HÌNH:
    - Chứa 3 thuật toán: Logistic Regression, Decision Tree, Random Forest.
    - class_weight='balanced': Rất quan trọng để xử lý dữ liệu lệch (ít người rời mạng).
    """
    return OrderedDict(
        {
            "Logistic Regression": Pipeline(
                steps=[
                    ("preprocess", build_preprocessor()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            solver="lbfgs",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "Decision Tree": Pipeline(
                steps=[
                    ("preprocess", build_preprocessor()),
                    (
                        "model",
                        DecisionTreeClassifier(
                            max_depth=6,
                            min_samples_leaf=25,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "Random Forest": Pipeline(
                steps=[
                    ("preprocess", build_preprocessor()),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=150,
                            max_depth=10,
                            min_samples_leaf=8,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        }
    )


def predict_binary_from_probability(probability: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """
    HÀM QUYẾT ĐỊNH NHÃN:
    - Nếu xác suất >= ngưỡng (ví dụ 0.5) thì là 1 (Churn), ngược lại là 0.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold phải nằm trong khoảng (0, 1).")
    return (probability >= threshold).astype(int)


def evaluate_classifier(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float]:
    """
    HÀM ĐÁNH GIÁ: Tính các chỉ số Accuracy, Precision, Recall, F1.
    """
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = predict_binary_from_probability(proba, threshold=threshold)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[Dict[str, Pipeline], pd.DataFrame]:
    trained_models: Dict[str, Pipeline] = {}
    rows = []
    for name, model in build_candidate_models().items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        metrics = evaluate_classifier(model, X_test, y_test)
        rows.append({"model": name, **metrics})

    metrics_df = pd.DataFrame(rows).sort_values(by="f1_score", ascending=False).reset_index(drop=True)
    return trained_models, metrics_df


def get_confusion_matrix(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = predict_binary_from_probability(proba, threshold=threshold)
    return confusion_matrix(y_test, y_pred, labels=[0, 1])


def evaluate_thresholds(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Compare several decision thresholds for business discussion."""
    if thresholds is None:
        thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

    rows = []
    proba = model.predict_proba(X_test)[:, 1]
    for threshold in thresholds:
        y_pred = predict_binary_from_probability(proba, threshold=threshold)
        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "predicted_churn_count": int(y_pred.sum()),
            }
        )
    return pd.DataFrame(rows)


def extract_feature_importance(model: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """Return top feature importances for models that expose feature_importances_."""
    classifier = model.named_steps["model"]
    if not hasattr(classifier, "feature_importances_"):
        raise TypeError("Mô hình hiện tại không hỗ trợ feature_importances_.")

    preprocessor = model.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def predict_churn_probability(
    model: Pipeline,
    customer_data: dict | pd.DataFrame,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Predict churn probability and label for one or more customers."""
    data = normalize_customer_input(customer_data)
    proba = model.predict_proba(data)[:, 1]
    pred = predict_binary_from_probability(proba, threshold=threshold)
    return pd.DataFrame(
        {
            "churn_probability": proba,
            "prediction": pred,
            "prediction_label": np.where(pred == 1, "Yes", "No"),
        }
    )
