from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# --- LOAD MODEL ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'models', 'robbery_model.pkl')
cols_path = os.path.join(base_dir, 'models', 'model_columns.pkl')

# Load these once when the app starts
model = joblib.load(model_path)
model_columns = joblib.load(cols_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Get data from form
            # Matches the input_features from 2_train_model.py
            premise = request.form['premisetype']
            division = request.form['division']
            hour = int(request.form['occurrencehour'])

            # 2. Create DataFrame
            input_data = pd.DataFrame([[premise, division, hour]], 
                                      columns=['premisetype', 'division', 'occurrencehour'])

            # 3. Preprocess (One Hot Encoding)
            input_encoded = pd.get_dummies(input_data)

            # 4. Align Columns (Crucial Step)
            # Add missing columns with 0, remove extra ones
            input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

            # 5. Predict
            prediction = model.predict(input_encoded)[0]

            return render_template('result.html', prediction=prediction)

        except Exception as e:
            return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)