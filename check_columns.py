import pandas as pd
import os

# Path to your cleaned data
base_dir = r"C:\Users\Admin\Desktop\Semester6\COMP309_DataWarehouse\FinalProject"
data_path = os.path.join(base_dir, 'data', 'robbery_cleaned.csv')

try:
    df = pd.read_csv(data_path)
    print("--- COLUMN NAMES ---")
    print(df.columns.tolist())
except Exception as e:
    print(f"Error: {e}")