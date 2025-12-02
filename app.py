from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# --- LOAD MODEL ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'models', 'robbery_model.pkl')
cols_path = os.path.join(base_dir, 'models', 'model_columns.pkl')

# Load model/columns safely
try:
    model = joblib.load(model_path)
    model_columns = joblib.load(cols_path)
    print("Model loaded successfully.")
except:
    print("WARNING: Model files not found. Run 2_train_model.py first.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Error: Model not loaded. Please train the model first."

    if request.method == 'POST':
        try:
            # 1. Get data from HTML Form
            # Note: These names come from the <input name="..."> in index.html
            premise = request.form['premise_input']
            division = request.form['division_input']
            hour = int(request.form['hour_input'])

            # 2. Create DataFrame with exact column names from Training
            # Columns must match the 'features' list in 2_train_model.py
            input_data = pd.DataFrame([[premise, division, hour]], 
                                      columns=['PREMISES_TYPE', 'DIVISION', 'OCC_HOUR'])

            # 3. One-Hot Encode the input
            input_encoded = pd.get_dummies(input_data)

            # 4. Align Columns (The most important step!)
            # This adds missing columns (with 0) and sorts them to match training data
            input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

            # 5. Predict
            prediction = model.predict(input_encoded)[0]

            return render_template('result.html', prediction=prediction)

        except Exception as e:
            return render_template('result.html', prediction=f"Error processing request: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)