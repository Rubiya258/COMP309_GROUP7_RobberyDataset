import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# --- SETUP PATHS ---
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'robbery_cleaned.csv')
model_folder = os.path.join(base_dir, 'models')
os.makedirs(model_folder, exist_ok=True)

print("--- Step 2: Model Training Started ---")

# 1. Load Data
df = pd.read_csv(data_path)

# 2. Feature Selection
# Let's pick features that are easy for a user to input in a web form.
# Based on your dataset columns
feature_cols = ['premisetype', 'offence', 'division', 'occurrenceyear'] 
# NOTE: 'offence' seems to be the target in your original script? 
# If you are predicting the TYPE of robbery, 'offence' is the target.
# If you are predicting if it is a MAJOR offense, we need a different target.
# Let's assume we are predicting 'offence' (Robbery Type) based on other factors.

target_col = 'offence' 
# We drop the target from features
input_features = ['premisetype', 'division', 'occurrencehour'] # Example features

# Ensure features exist
df = df.dropna(subset=input_features + [target_col])

X = df[input_features]
y = df[target_col]

# 3. Preprocessing for Model
# Convert categorical variables (One Hot Encoding)
X_encoded = pd.get_dummies(X)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# 5. Train Model (Decision Tree)
print("Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

score = dt_model.score(X_test, y_test)
print(f"Model Accuracy: {score:.2f}")

# 6. Save Model and Columns
# We MUST save the columns to align the Flask input later
joblib.dump(dt_model, os.path.join(model_folder, 'robbery_model.pkl'))
joblib.dump(list(X_encoded.columns), os.path.join(model_folder, 'model_columns.pkl'))

print("Model and columns saved to /models folder.")