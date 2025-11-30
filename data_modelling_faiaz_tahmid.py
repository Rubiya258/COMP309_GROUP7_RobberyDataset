
# -*- coding: utf-8 -*-
"""
COMP 309 - Data Warehousing & Predictive Analytics
Group Project - Toronto Robbery Open Data

SECTION: DATA MODELLING
Author: Faiaz Tahmid

This script includes:
- Data transformations (cleaning, missing data, encoding, scaling)
- Feature selection using SelectKBest
- Train/Test split (stratified)
- Class imbalance handling
- Predictive model building (Logistic Regression, Decision Tree)
- Model evaluation (accuracy, classification report, confusion matrix, ROC/AUC)
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")
sns.set_theme()

print("DATA MODELLING SECTION - Author: Faiaz Tahmid")
print("=" * 60)

# Load dataset (adjust path if needed)
csv_path = r"Robbery_Open_Data.csv"
df = pd.read_csv(csv_path)
print("\nLoaded dataset with shape:", df.shape)

# Target creation
df["TARGET_COMMERCIAL_PREMISES"] = np.where(
    df["PREMISES_TYPE"] == "Commercial", 1, 0
)
print("\nTarget distribution:")
print(df["TARGET_COMMERCIAL_PREMISES"].value_counts(normalize=True))

# Drop identifiers
drop_cols = [
    "OBJECTID", "EVENT_UNIQUE_ID", "REPORT_DATE", "OCC_DATE",
    "PREMISES_TYPE", "x", "y"
]
drop_cols = [c for c in drop_cols if c in df.columns]
df_model = df.drop(columns=drop_cols)

# Split features/target
y = df_model["TARGET_COMMERCIAL_PREMISES"]
X = df_model.drop(columns=["TARGET_COMMERCIAL_PREMISES"])

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTrain/Test split completed.")

# Feature Selection
feature_selector = SelectKBest(
    score_func=mutual_info_classif,
    k=30
)

# Model Pipelines
log_reg_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("select", feature_selector),
    ("classifier", LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced"
    ))
])

dt_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("select", feature_selector),
    ("classifier", DecisionTreeClassifier(
        class_weight="balanced",
        min_samples_split=10,
        random_state=42
    ))
])

# Train models
print("\nTraining Logistic Regression...")
log_reg_pipeline.fit(X_train, y_train)

print("Training Decision Tree...")
dt_pipeline.fit(X_train, y_train)

# Evaluation helper
def evaluate_model(name, model, X_test, y_test):
    print("\n" + "=" * 60)
    print(f"MODEL EVALUATION: {name}")
    print("=" * 60)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min())

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{name} - ROC Curve")
    plt.legend()
    plt.show()

    print(f"ROC AUC Score: {auc:.4f}")

# Score models
evaluate_model("Logistic Regression", log_reg_pipeline, X_test, y_test)
evaluate_model("Decision Tree Classifier", dt_pipeline, X_test, y_test)

print("\nDATA MODELLING SECTION COMPLETED ")
