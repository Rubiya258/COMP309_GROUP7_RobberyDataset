# ============================================================
# DATA MODELLING SECTION
# COMP 309 - Data Warehousing & Predictive Analytics
# Group Project - Toronto Robbery Open Data
# Author: Faiaz Tahmid
# ============================================================

\"\"\"
This section performs the following tasks:

- Data transformations:
      • Removing invalid coordinates (LAT/LONG = 0)
      • Handling missing values
      • Encoding categorical features
      • Standardizing numerical features

- Feature selection:
      • SelectKBest using mutual information

- Train/Test Split:
      • 80/20 split with class stratification

- Handling Imbalanced Classes:
      • Using class_weight="balanced"

- Predictive modelling:
      • Logistic Regression
      • Decision Tree Classifier

- Model evaluation:
      • Accuracy
      • Classification report
      • Confusion matrix (heatmap)
      • ROC curve and AUC score

- Model selection:
      • Recommending best model based on AUC
\"\"\"

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

plt.style.use(\"default\")
sns.set_theme()

print(\"\\n==============================\")
print(\"DATA MODELLING STAGE STARTED\")
print(\"Author: Faiaz Tahmid\")
print(\"==============================\\n\")

# ------------------------------------------------------------
# 1. LOAD DATASET OR REUSE EXISTING df
# ------------------------------------------------------------

try:
    df  # type: ignore[name-defined]
    print(\"Using dataframe 'df' from Data Exploration section.\\n\")
except NameError:
    print(\"No dataframe found from exploration. Loading CSV...\\n\")
    csv_path = r\"Robbery_Open_Data.csv\"
    df = pd.read_csv(csv_path)
    print(\"Loaded dataset with shape:\", df.shape)

# ------------------------------------------------------------
# 2. DATA CLEANING & PREPARATION
# ------------------------------------------------------------

# Drop identifier columns that do not help with prediction
drop_cols = [
    \"OBJECTID\",
    \"EVENT_UNIQUE_ID\",
    \"REPORT_DATE\",
    \"OCC_DATE\",
    \"x\",
    \"y\"
]
drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols)

# Remove invalid coordinate rows (identified during exploration)
if \"LAT_WGS84\" in df.columns and \"LONG_WGS84\" in df.columns:
    before = df.shape[0]
    df = df[(df[\"LAT_WGS84\"] != 0) & (df[\"LONG_WGS84\"] != 0)]
    after = df.shape[0]
    print(f\"Removed {before - after} rows with invalid LAT/LONG = 0\\n\")

# ------------------------------------------------------------
# 3. TARGET VARIABLE CREATION
# ------------------------------------------------------------

if \"PREMISES_TYPE\" not in df.columns:
    raise ValueError(\"PREMISES_TYPE column missing from dataset.\")

df[\"TARGET_COMMERCIAL_PREMISES\"] = (df[\"PREMISES_TYPE\"] == \"Commercial\").astype(int)

print(\"Target distribution:\")
print(df[\"TARGET_COMMERCIAL_PREMISES\"].value_counts(), \"\\n\")
print(\"Target distribution (%):\")
print(df[\"TARGET_COMMERCIAL_PREMISES\"].value_counts(normalize=True), \"\\n\")

# ------------------------------------------------------------
# 4. FEATURE/TARGET SPLIT
# ------------------------------------------------------------

target = \"TARGET_COMMERCIAL_PREMISES\"
exclude_from_features = [\"PREMISES_TYPE\", target]

exclude_from_features = [c for c in exclude_from_features if c in df.columns]

y = df[target]
X = df.drop(columns=exclude_from_features)

print(\"X shape:\", X.shape)
print(\"y shape:\", y.shape, \"\\n\")

# Identify numeric & categorical columns
numeric_cols = X.select_dtypes(include=[\"int64\", \"float64\"]).columns.tolist()
categorical_cols = X.select_dtypes(include=[\"object\"]).columns.tolist()

print(\"Numeric columns:\", numeric_cols)
print(\"\\nCategorical columns:\", categorical_cols, \"\\n\")

# ------------------------------------------------------------
# 5. PREPROCESSING PIPELINES
# ------------------------------------------------------------

numeric_pipeline = Pipeline(steps=[
    (\"imputer\", SimpleImputer(strategy=\"median\")),
    (\"scaler\", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    (\"imputer\", SimpleImputer(strategy=\"most_frequent\")),
    (\"encoder\", OneHotEncoder(handle_unknown=\"ignore\"))
])

preprocessor = ColumnTransformer(
    transformers=[
        (\"num\", numeric_pipeline, numeric_cols),
        (\"cat\", categorical_pipeline, categorical_cols)
    ]
)

# ------------------------------------------------------------
# 6. TRAIN/TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(\"Train-test split completed:\")
print(\"X_train:\", X_train.shape)
print(\"X_test :\", X_test.shape, \"\\n\")

# ------------------------------------------------------------
# 7. FEATURE SELECTION
# ------------------------------------------------------------

selector = SelectKBest(score_func=mutual_info_classif, k=30)

# ------------------------------------------------------------
# 8. MODEL PIPELINES
# ------------------------------------------------------------

log_reg_model = Pipeline(steps=[
    (\"preprocess\", preprocessor),
    (\"select\", selector),
    (\"classifier\", LogisticRegression(
        class_weight=\"balanced\",
        max_iter=1000,
        solver=\"lbfgs\"
    ))
])

tree_model = Pipeline(steps=[
    (\"preprocess\", preprocessor),
    (\"select\", selector),
    (\"classifier\", DecisionTreeClassifier(
        class_weight=\"balanced\",
        min_samples_split=10,
        random_state=42
    ))
])

# ------------------------------------------------------------
# 9. TRAIN MODELS
# ------------------------------------------------------------

print(\"Training Logistic Regression...\")
log_reg_model.fit(X_train, y_train)

print(\"Training Decision Tree...\")
tree_model.fit(X_train, y_train)

# ------------------------------------------------------------
# 10. EVALUATION FUNCTION
# ------------------------------------------------------------

def evaluate_model(name, model, X_test, y_test):
    print(\"\\n\" + \"=\" * 70)
    print(f\"EVALUATION: {name}\")
    print(\"=\" * 70 + \"\\n\")

    y_pred = model.predict(X_test)

    # Probabilities for ROC curve
    if hasattr(model, \"predict_proba\"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min())

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f\"Accuracy: {acc:.4f}\\n\")

    print(\"Classification Report:\")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[\"Actual 0 (Non-Commercial)\", \"Actual 1 (Commercial)\"],
        columns=[\"Pred 0\", \"Pred 1\"]
    )
    print(\"\\nConfusion Matrix:\")
    print(cm_df, \"\\n\")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, cmap=\"Blues\", fmt=\"d\")
    plt.title(f\"{name} - Confusion Matrix\")
    plt.tight_layout()
    plt.show()

    # ROC Curve + AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    print(f\"AUC Score: {auc:.4f}\\n\")

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f\"AUC = {auc:.3f}\")
    plt.plot([0, 1], [0, 1], linestyle=\"--\")
    plt.title(f\"{name} - ROC Curve\")
    plt.xlabel(\"FPR\")
    plt.ylabel(\"TPR\")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return acc, auc

# ------------------------------------------------------------
# 11. RUN EVALUATION
# ------------------------------------------------------------

log_acc, log_auc = evaluate_model(\"Logistic Regression\", log_reg_model, X_test, y_test)
tree_acc, tree_auc = evaluate_model(\"Decision Tree Classifier\", tree_model, X_test, y_test)

# ------------------------------------------------------------
# 12. MODEL SELECTION SUMMARY
# ------------------------------------------------------------

print(\"\\n==============================\")
print(\"MODEL COMPARISON SUMMARY\")
print(\"==============================\")

print(f\"Logistic Regression: Accuracy = {log_acc:.4f}, AUC = {log_auc:.4f}\")
print(f\"Decision Tree:       Accuracy = {tree_acc:.4f}, AUC = {tree_auc:.4f}\")

best_model = \"Logistic Regression\" if log_auc > tree_auc else \"Decision Tree Classifier\"

print(f\"\\nBest Performing Model Based on AUC: {best_model}\")

print(\"\\nDATA MODELLING SECTION COMPLETED — Author: Faiaz Tahmid\\n\")
