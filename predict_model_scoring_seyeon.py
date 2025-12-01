# ============================================================
# PREDICTIVE MODEL BUILDING & MODEL SCORING AND EVALUATION
# COMP 309 - Group Project 2
# Author: Seyeon Jo
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score, top_k_accuracy_score
)
from sklearn.base import clone

# -----------------------------
# 0) CONFIG
# -----------------------------
TARGET_MODE = "binary"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_CLASS_COUNT = 100
OUT_DIR = "outputs"

plt.style.use("default")
sns.set_theme()
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 1) LOAD
# -----------------------------
DATA_DIR = r"C:\\Users\\seyeo\\PycharmProjects\\PythonProject\\group_proj2"
CSV_PATH = os.path.join(DATA_DIR, "Robbery_Open_Data.csv")
df = pd.read_csv(CSV_PATH)
print("Loaded dataset:", df.shape)

# -----------------------------
# 2) BASIC CLEANING
# -----------------------------
if {'LAT_WGS84', 'LONG_WGS84'}.issubset(df.columns):
    before = len(df)
    df = df[(df['LAT_WGS84'] != 0) & (df['LONG_WGS84'] != 0)]
    print(f"Removed {before - len(df)} rows with invalid LAT/LONG = 0")

# -----------------------------
# 3) TARGET DEFINITION
# -----------------------------
if TARGET_MODE == "binary":
    if "PREMISES_TYPE" not in df.columns:
        raise ValueError("PREMISES_TYPE column missing for binary mode.")
    df["TARGET"] = (df["PREMISES_TYPE"] == "Commercial").astype(int)
    base_drop = ["PREMISES_TYPE"]
else:
    if "UCR_EXT" not in df.columns:
        raise ValueError("UCR_EXT column missing for multiclass mode.")
    df["UCR_EXT"] = df["UCR_EXT"].astype(str)
    vc = df["UCR_EXT"].value_counts()
    rare = vc[vc < MIN_CLASS_COUNT].index
    df.loc[df["UCR_EXT"].isin(rare), "UCR_EXT"] = "Other"
    df["TARGET"] = df["UCR_EXT"]
    base_drop = ["UCR_EXT"]

base_drop += ["OFFENCE", "OBJECTID", "EVENT_UNIQUE_ID", "REPORT_DATE", "OCC_DATE", "x", "y"]

# -----------------------------
# 3-1) REMOVE KNOWN LEAKAGE COLUMNS
# -----------------------------
leak_candidates_manual = [
    "PREMISES_TYPE_DESC", "PREMISES_TYPE_CATEGORY",
    "OCC_LOCATION", "LOCATION_TYPE", "LOCATION_CATEGORY",
    "UCR_CODE", "UCR_CODE_GROUP", "OCC_PREMISES_TYPE"
]

drop_cols = [c for c in (base_drop + leak_candidates_manual) if c in df.columns]

y = df["TARGET"].copy()
tmp_X = df.drop(columns=drop_cols + ["TARGET"], errors="ignore")

# -----------------------------
# 3-2) AUTO LEAKAGE DETECTION
# -----------------------------
auto_leaks = []
for col in tmp_X.select_dtypes(include=["object"]).columns:
    tmp_join = tmp_X[[col]].copy()
    tmp_join["TARGET"] = y.values
    nunique_per_value = tmp_join.groupby(col)["TARGET"].nunique(dropna=False)
    if nunique_per_value.max() == 1:
        auto_leaks.append(col)

if auto_leaks:
    print(f"[Leakage detected & removed] {auto_leaks}")

X = tmp_X.drop(columns=auto_leaks, errors="ignore")

print("\nTarget distribution:")
print(y.value_counts())

# -----------------------------
# 4) SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print("\nTrain:", X_train.shape, " Test:", X_test.shape)

# -----------------------------
# 5) PREPROCESSOR
# -----------------------------
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

try:
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
except TypeError:
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# -----------------------------
# 6) FEATURE SELECTION
# -----------------------------
pre_tmp = clone(pre).fit(X_train, y_train)
total_feats = pre_tmp.transform(X_train.iloc[:5]).shape[1]
k_desired = 40 if TARGET_MODE == "multiclass" else 30
k_safe = max(1, min(k_desired, total_feats - 1))
print(f"\nSelectKBest: k={k_safe} / total_features={total_feats}")

# -----------------------------
# 7) MODELS
# -----------------------------
selector_lr = SelectKBest(mutual_info_classif, k=k_safe)
selector_dt = SelectKBest(mutual_info_classif, k=k_safe)

logi = Pipeline([
    ("pre", clone(pre)),
    ("sel", selector_lr),
    ("clf", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial" if TARGET_MODE == "multiclass" else "auto"
    ))
])

tree = Pipeline([
    ("pre", clone(pre)),
    ("sel", selector_dt),
    ("clf", DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    ))
])

# -----------------------------
# 8) TRAIN
# -----------------------------
print("\nTraining Logistic Regression...")
logi.fit(X_train, y_train)
print("Training Decision Tree...")
tree.fit(X_train, y_train)

# -----------------------------
# 9) EVALUATION
# -----------------------------
def save_cm(name, y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7, 5))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"cm_{name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def eval_binary(name, model):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cr = classification_report(y_test, y_pred, digits=4)

    print(f"\n=== {name} (Binary) ===")
    print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    print(cr)

    cm_path = save_cm(name, y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1], [0,1], "--")
    plt.legend()
    plt.title(f"{name} - ROC Curve"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.tight_layout()
    roc_path = os.path.join(OUT_DIR, f"roc_{name.replace(' ', '_')}.png")
    plt.savefig(roc_path, dpi=150); plt.close()

    with open(os.path.join(OUT_DIR, f"report_{name.replace(' ', '_')}.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== {name} (Binary) ===\nAccuracy: {acc:.4f} | AUC: {auc:.4f}\n\n")
        f.write(cr)

    return {"acc": acc, "auc": auc, "cm": cm_path, "roc": roc_path}

# -----------------------------
# 10) RUN EVAL
# -----------------------------
r1 = eval_binary("Logistic Regression", logi)
r2 = eval_binary("Decision Tree", tree)

plt.figure()
for model, name in [(logi, "Logistic Regression"), (tree, "Decision Tree")]:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})")
plt.plot([0,1], [0,1], "--")
plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_comparison.png"), dpi=150)
plt.close()

best = "Logistic Regression" if r1["auc"] >= r2["auc"] else "Decision Tree"
results = pd.DataFrame([
    {"Model": "Logistic Regression", "Accuracy": r1["acc"], "AUC": r1["auc"]},
    {"Model": "Decision Tree", "Accuracy": r2["acc"], "AUC": r2["auc"]}
])
results.to_csv(os.path.join(OUT_DIR, "model_scores.csv"), index=False)
print("\nSummary:")
print(results)
print(f"\n✅ Best Model: {best}")
print(f"📁 Results saved under: {os.path.abspath(OUT_DIR)}")
