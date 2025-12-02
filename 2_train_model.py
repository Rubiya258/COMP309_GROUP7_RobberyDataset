import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# --- SETUP PATHS ---
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'robbery_cleaned.csv')
model_folder = os.path.join(base_dir, 'models')
os.makedirs(model_folder, exist_ok=True)

print("--- Step 2: Model Training Started ---")

try:
    # 1. Load Cleaned Data
    df = pd.read_csv(data_path)
    
    # 2. Separate Features (X) and Target (y)
    # Target: OFFENCE (What we want to predict)
    # Features: PREMISES_TYPE, DIVISION, OCC_HOUR
    
    target_col = 'OFFENCE'
    features = ['PREMISES_TYPE', 'DIVISION', 'OCC_HOUR']
    
    X = df[features]
    y = df[target_col]

    # 3. One-Hot Encoding
    # This converts text like 'Outside' to numbers 0/1
    X_encoded = pd.get_dummies(X)
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

    # 5. Train Model
    print("Training Decision Tree Model...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Score
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2%}")

    # 6. Save Model AND Column List
    # Saving the column list is CRITICAL for Flask to work later
    joblib.dump(model, os.path.join(model_folder, 'robbery_model.pkl'))
    joblib.dump(list(X_encoded.columns), os.path.join(model_folder, 'model_columns.pkl'))
    
    print("Success! Model and columns saved to /models folder.")

except Exception as e:
    print(f"Error: {e}")