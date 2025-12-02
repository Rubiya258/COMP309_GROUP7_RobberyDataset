# Robbery Open Data Analysis & Prediction - Group 7

This project analyzes Robbery Open Data to identify patterns and predicts the type of offence based on location and time.

## 📂 Project Structure
* **data/**: Contains the raw and processed dataset.
* **models/**: Stores the trained `.pkl` models.
* **templates/**: HTML files for the Flask web interface.
* **1_preprocess.py**: Cleans the data and prepares it for training.
* **2_train_model.py**: Trains the Decision Tree classifier and saves the model.
* **app.py**: The Flask web application.

## 🚀 How to Run
1. **Install Dependencies**:
pip install pandas numpy scikit-learn flask joblib
py -m pip install pandas numpy scikit-learn flask joblib
2. **Prepare Data**:
Run the preprocessing script to clean the data.

python 1_preprocess.py
py 1_preprocess.py
3. **Train Model**:
Run the training script to generate the model files in the `/models` folder.

python 2_train_model.py
py 2_train_model.py

4. **Start Web Server**:
Run the Flask app.

python app.py
py 2_train_model.py

Open your browser and go to `http://127.0.0.1:5000/`.

## 👥 Contributors
* Faiaz & Tahmid (Data Modelling)
* Seyeon (Model Scoring)
* Van Nguyen (Deployment & Integration)