import pandas as pd
import os

# --- SETUP PATHS ---
base_dir = os.path.dirname(os.path.abspath(__file__))
# We assume raw data is in a 'data' subfolder or the same folder
data_path = os.path.join(base_dir, 'data', 'Robbery_Open_Data.csv')
output_path = os.path.join(base_dir, 'data', 'robbery_cleaned.csv')

# Create data directory if it doesn't exist
os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)

print("--- Step 1: Preprocessing Started ---")

# 1. Load Data
# (If you get a file not found error, ensure your csv is in the /data folder)
if not os.path.exists(data_path):
    print(f"Warning: Raw file not found at {data_path}. Checking current directory...")
    data_path = os.path.join(base_dir, 'Robbery_Open_Data.csv')

try:
    df = pd.read_csv(data_path)
    print(f"Raw data loaded. Shape: {df.shape}")

    # 2. Feature Selection (Keep only what we need + Target)
    # Based on your file analysis, these are the correct UPPERCASE names
    required_cols = ['PREMISES_TYPE', 'DIVISION', 'OCC_HOUR', 'OFFENCE']
    
    # Check if columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    df_clean = df[required_cols].copy()

    # 3. Clean Missing Values
    # Drop rows where any of our critical features are missing
    df_clean.dropna(inplace=True)
    
    # 4. Save
    df_clean.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
    print(f"Final shape: {df_clean.shape}")

except Exception as e:
    print(f"Error during preprocessing: {e}")