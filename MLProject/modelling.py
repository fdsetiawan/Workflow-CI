"""
Baseline modelling script for Dicoding MSML final project.

Author:
Fauzi Dwi Setiawan

Dataset:
Diabetes Prediction Dataset

Folder structure
----------------
Workflow-CI/
└── MLProject/
    ├── diabetes_prediction_preprocessed.csv
    ├── modelling.py
    ├── MLproject
    └── conda.yaml

Important note
--------------
The preprocessing stage before this script only performs data cleaning.
Train-test split, feature scaling/standardization, and categorical encoding
are performed in this modelling script to avoid data leakage.

Purpose
-------
- Load cleaned dataset: diabetes_prediction_preprocessed.csv.
- Split data into train and test sets.
- Apply modelling-stage preprocessing:
  - StandardScaler for continuous numeric features.
  - OneHotEncoder for categorical features.
  - Passthrough for binary numeric features.
- Train baseline Random Forest model.
- Evaluate model and save artifacts.
- Track experiment using MLflow.

How to run
----------
From inside the Workflow-CI/MLProject folder:

    python modelling.py

Open MLflow UI:

    mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "diabetes"
RANDOM_STATE = 42
TEST_SIZE = 0.20

CONTINUOUS_FEATURES = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

BINARY_FEATURES = [
    "hypertension",
    "heart_disease",
]

CATEGORICAL_FEATURES = [
    "gender",
    "smoking_history",
]


# -----------------------------------------------------------------------------
# Argument, data loading, and validation utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline diabetes prediction modelling with MLflow tracking."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="diabetes_prediction_preprocessed.csv",
        help="Path to cleaned/preprocessed dataset. Default assumes this script is run inside Workflow-CI/MLProject.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Diabetes_Prediction_Baseline",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="file:./mlruns",
        help="MLflow tracking URI. Local default: file:./mlruns.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="artifacts_baseline",
        help="Local directory to save generated artifacts.",
    )
    return parser.parse_args()


def load_dataset(dataset_path: str) -> pd.DataFrame:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_file.resolve()}\n"
            "Make sure diabetes_prediction_preprocessed.csv is in the Workflow-CI/MLProject folder, "
            "or pass the correct path using --dataset-path."
        )

    df = pd.read_csv(dataset_file)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    required_columns = (
        [TARGET_COLUMN]
        + CONTINUOUS_FEATURES
        + BINARY_FEATURES
        + CATEGORICAL_FEATURES
    )
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# -----------------------------------------------------------------------------
# Modelling-stage preprocessing and baseline model
# -----------------------------------------------------------------------------


def make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder compatible with older and newer scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def create_preprocessor() -> ColumnTransformer:
    """
    Create preprocessing transformer.

    This step is intentionally placed in modelling.py because the earlier
    preprocessing stage only performs cleaning. The transformer is fitted only
    on X_train to avoid data leakage.
    """
    return ColumnTransformer(
        transformers=[
            ("standard_scaler", StandardScaler(), CONTINUOUS_FEATURES),
            ("one_hot_encoder", make_one_hot_encoder(), CATEGORICAL_FEATURES),
            ("binary_passthrough", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def create_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def create_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", create_model()),
        ]
    )


def get_feature_names(pipeline: Pipeline) -> List[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Safe fallback if the environment uses an older scikit-learn version.
        categorical_features = list(
            preprocessor.named_transformers_["one_hot_encoder"].get_feature_names_out(
                CATEGORICAL_FEATURES
            )
        )
        return CONTINUOUS_FEATURES + categorical_features + BINARY_FEATURES


# -----------------------------------------------------------------------------
# Evaluation and artifact utilities
# -----------------------------------------------------------------------------


def evaluate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    prefix: str,
) -> Dict[str, float]:
    y_pred = pipeline.predict(X)
    metrics = {
        f"{prefix}_accuracy": float(accuracy_score(y, y_pred)),
        f"{prefix}_precision": float(precision_score(y, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y, y_pred, zero_division=0)),
        f"{prefix}_f1_score": float(f1_score(y, y_pred, zero_division=0)),
    }

    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X)[:, 1]
        metrics[f"{prefix}_roc_auc"] = float(roc_auc_score(y, y_proba))

    return metrics


def save_json(data: Dict[str, object], output_path: Path) -> Path:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
    return output_path


def save_classification_report(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifact_dir: Path,
) -> Path:
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    output_path = artifact_dir / "classification_report.txt"
    output_path.write_text(report, encoding="utf-8")
    return output_path


def save_confusion_matrix_plot(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifact_dir: Path,
) -> Path:
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm)
    ax.set_title("Confusion Matrix - Baseline Random Forest")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Diabetes", "Diabetes"])
    ax.set_yticklabels(["No Diabetes", "Diabetes"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    output_path = artifact_dir / "confusion_matrix.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_roc_curve_plot(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifact_dir: Path,
) -> Path:
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve - Baseline Random Forest")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()

    output_path = artifact_dir / "roc_curve.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_feature_importance_plot(
    pipeline: Pipeline,
    artifact_dir: Path,
) -> Path:
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    importance_df = pd.DataFrame(
        {
            "feature": sorted_features,
            "importance": sorted_importances,
        }
    )
    csv_path = artifact_dir / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_features[::-1], sorted_importances[::-1])
    ax.set_title("Feature Importance - Baseline Random Forest")
    ax.set_xlabel("Importance")
    fig.tight_layout()

    output_path = artifact_dir / "feature_importance.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(args.tracking_uri)

    # When this script is executed by `mlflow run .`, MLflow has already
    # created a run and stores its ID in MLFLOW_RUN_ID. In that case, do not
    # call set_experiment() because it can point the script to a different
    # experiment and trigger: "active run ID does not match environment run ID".
    mlflow_project_run_id = os.environ.get("MLFLOW_RUN_ID")
    run_name = "baseline_random_forest_with_modelling_preprocessing"

    if mlflow_project_run_id is None:
        # Direct execution, for example: python modelling.py
        mlflow.set_experiment(args.experiment_name)
        run_context = mlflow.start_run(run_name=run_name)
    else:
        # Execution through MLproject, for example: mlflow run . --env-manager local
        run_context = mlflow.start_run(run_id=mlflow_project_run_id)

    df = load_dataset(args.dataset_path)
    validate_required_columns(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    with run_context as run:
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.set_tag("execution_mode", "mlflow_project" if mlflow_project_run_id else "direct_python")
        pipeline = create_pipeline()
        pipeline.fit(X_train, y_train)

        train_metrics = evaluate_model(pipeline, X_train, y_train, prefix="train")
        test_metrics = evaluate_model(pipeline, X_test, y_test, prefix="test")
        all_metrics = {**train_metrics, **test_metrics}

        params = {
            "model_type": "RandomForestClassifier",
            "target_column": TARGET_COLUMN,
            "dataset_path": args.dataset_path,
            "data_rows": int(df.shape[0]),
            "data_columns": int(df.shape[1]),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "continuous_features": ",".join(CONTINUOUS_FEATURES),
            "binary_features": ",".join(BINARY_FEATURES),
            "categorical_features": ",".join(CATEGORICAL_FEATURES),
            "preprocessing_in_modelling": True,
            "scaler": "StandardScaler",
            "encoder": "OneHotEncoder(handle_unknown='ignore')",
        }

        model_params = pipeline.named_steps["model"].get_params()
        for key in [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "class_weight",
        ]:
            params[f"rf_{key}"] = model_params[key]

        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in all_metrics.items():
            mlflow.log_metric(key, value)

        model_path = artifact_dir / "baseline_diabetes_pipeline.joblib"
        joblib.dump(pipeline, model_path)

        metrics_path = save_json(all_metrics, artifact_dir / "metrics_summary.json")
        params_path = save_json(params, artifact_dir / "params_summary.json")
        report_path = save_classification_report(pipeline, X_test, y_test, artifact_dir)
        cm_path = save_confusion_matrix_plot(pipeline, X_test, y_test, artifact_dir)
        roc_path = save_roc_curve_plot(pipeline, X_test, y_test, artifact_dir)
        fi_path = save_feature_importance_plot(pipeline, artifact_dir)
        fi_csv_path = artifact_dir / "feature_importance.csv"

        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        mlflow.log_artifact(str(model_path), artifact_path="model_pickle")
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        mlflow.log_artifact(str(params_path), artifact_path="evaluation")
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")
        mlflow.log_artifact(str(cm_path), artifact_path="figures")
        mlflow.log_artifact(str(roc_path), artifact_path="figures")
        mlflow.log_artifact(str(fi_path), artifact_path="figures")
        mlflow.log_artifact(str(fi_csv_path), artifact_path="explainability")

        print("Baseline training completed successfully.")
        print(f"MLflow Run ID: {run.info.run_id}")
        print("Test metrics:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
