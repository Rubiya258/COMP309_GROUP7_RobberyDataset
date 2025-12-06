# -*- coding: utf-8 -*-
"""
OFFENCE Target Prediction Test
Purpose:
- Test whether OFFENCE can be predicted from:
    ["PREMISES_TYPE", "DIVISION", "OCC_HOUR"]
- This is the mathematical justification for selecting a different target.
Author: Faiaz Tahmid
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# -----------------------------
# 1. Load Dataset
# -----------------------------
print("Loading dataset...")
df = pd.read_csv("Robbery_Open_Data.csv")
print("Dataset shape:", df.shape, "\n")

# -----------------------------
# 2. Clean invalid coordinates
# -----------------------------
df = df[(df["LAT_WGS84"] != 0) & (df["LONG_WGS84"] != 0)]

# -----------------------------
# 3. Select target & features
# -----------------------------
target = "OFFENCE"
features = ["PREMISES_TYPE", "DIVISION", "OCC_HOUR"]

df_model = df[features + [target]].dropna()

X = df_model[features]
y = df_model[target]

print("Unique OFFENCE categories:", y.nunique())
print("Sample size for modelling:", df_model.shape, "\n")

# -----------------------------
# 4. Preprocessing
# -----------------------------
categorical_features = ["PREMISES_TYPE", "DIVISION"]
numeric_features = ["OCC_HOUR"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# -----------------------------
# 5. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -----------------------------
# 6. Model: Logistic Regression
# -----------------------------
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

print("Training OFFENCE classifier...")
model.fit(X_train, y_train)

# -----------------------------
# 7. Evaluation
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nACCURACY WHEN PREDICTING OFFENCE:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nNOTE:")
print("- If accuracy is low, OFFENCE is NOT predictable from these features.")
print("- This mathematically justifies switching to a binary target like COMMERCIAL vs NON-COMMERCIAL.")
