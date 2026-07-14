"""
NovaTrust Credit Risk Intelligence Platform

Phase 2
Machine Learning Model Training

Author: Sakshi Patil
"""

import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

print("="*60)
print("NovaTrust Machine Learning Pipeline")
print("="*60)

# Locate project root
project_root = Path(__file__).resolve().parents[2]

# Path to cleaned dataset
dataset_path = (
    project_root
    / "01_data_engineering"
    / "processed_data"
    / "clean_loans.csv"
)

print("\nLoading cleaned dataset...")

df = pd.read_csv(dataset_path)

print("Dataset Loaded Successfully!")

print(f"Rows : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nColumn Names")
print("-" * 60)

for column in df.columns:
    print(column)

print("\nPreparing Features and Target")
print("-" * 60)

# Target Variable
y = df["Status"]

# Features
X = df.drop(columns=["Status"])

print(f"Features Shape : {X.shape}")
print(f"Target Shape   : {y.shape}")

print("\nIdentifying Categorical Columns")
print("-" * 60)

categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()

print(f"Number of Categorical Columns : {len(categorical_columns)}")

print("\nCategorical Columns:")

for column in categorical_columns:
    print(column)

print("\nSplitting Dataset")
print("-" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training Features : {X_train.shape}")
print(f"Testing Features  : {X_test.shape}")

print(f"Training Target   : {y_train.shape}")
print(f"Testing Target    : {y_test.shape}")

print("\nBuilding Logistic Regression Pipeline")
print("-" * 60)

# Identify categorical columns
categorical_columns = X.select_dtypes(include=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

# Pipeline
logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]
)

print("Training Logistic Regression...")

logistic_pipeline.fit(X_train, y_train)

print("Training Complete!")

print("\nEvaluating Logistic Regression Model")
print("-" * 60)

# Make predictions
y_pred = logistic_pipeline.predict(X_test)

# Predict probabilities
y_prob = logistic_pipeline.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\n")
print("=" * 60)
print("Logistic Regression with SMOTE")
print("=" * 60)

smote_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(max_iter=1000))
    ]
)

print("Training Logistic Regression with SMOTE...")

smote_pipeline.fit(X_train, y_train)

print("Training Complete!")

# Predictions
y_pred_smote = smote_pipeline.predict(X_test)
y_prob_smote = smote_pipeline.predict_proba(X_test)[:, 1]

print("\nEvaluation")

print(f"Accuracy : {accuracy_score(y_test, y_pred_smote):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_smote):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_smote):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred_smote):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, y_prob_smote):.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_smote))

print("\n")
print("=" * 60)
print("Random Forest Model")
print("=" * 60)

rf_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("Training Random Forest...")

rf_pipeline.fit(X_train, y_train)

print("Training Complete!")

# Predictions
y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

print("\nEvaluation")

print(f"Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, y_prob_rf):.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_rf))

print("\n")
print("=" * 60)
print("XGBoost Model")
print("=" * 60)

xgb_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        (
            "model",
            XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss"
            )
        )
    ]
)

print("Training XGBoost...")

xgb_pipeline.fit(X_train, y_train)

print("Training Complete!")

# Predictions
y_pred_xgb = xgb_pipeline.predict(X_test)
y_prob_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]

print("\nEvaluation")

print(f"Accuracy : {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_xgb):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred_xgb):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, y_prob_xgb):.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_xgb))

print("\nSaving Best Model")
print("-" * 60)

model_folder = (
    project_root
    / "02_machine_learning"
    / "models"
)

model_folder.mkdir(exist_ok=True)

model_path = model_folder / "final_model.pkl"

joblib.dump(xgb_pipeline, model_path)

print(f"Model saved to:\n{model_path}")

print("\nGenerating Risk Scores")
print("-" * 60)

# Score every loan
risk_scores = xgb_pipeline.predict_proba(X)[:, 1]

# Add predictions
scored_df = df.copy()

scored_df["risk_score"] = risk_scores

# Create risk bands
def assign_band(score):
    if score < 0.30:
        return "Low"
    elif score < 0.70:
        return "Medium"
    else:
        return "High"

scored_df["risk_band"] = scored_df["risk_score"].apply(assign_band)

print(scored_df[["risk_score", "risk_band"]].head())

print("\nSaving Scored Dataset")
print("-" * 60)

output_folder = (
    project_root
    / "02_machine_learning"
    / "outputs"
)

output_folder.mkdir(exist_ok=True)

output_path = output_folder / "scored_loans.csv"

scored_df.to_csv(output_path, index=False)

print(f"Scored dataset saved to:\n{output_path}")

print("\nSaving Model Comparison")
print("-" * 60)

metrics_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Logistic Regression + SMOTE",
        "Random Forest",
        "XGBoost"
    ],
    "ROC_AUC": [
        roc_auc_score(y_test, y_prob),
        roc_auc_score(y_test, y_prob_smote),
        roc_auc_score(y_test, y_prob_rf),
        roc_auc_score(y_test, y_prob_xgb)
    ]
})

metrics_path = output_folder / "model_metrics.csv"

metrics_df.to_csv(metrics_path, index=False)

print(metrics_df)

print(f"\nMetrics saved to:\n{metrics_path}")

