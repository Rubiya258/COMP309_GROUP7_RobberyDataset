import pandas as pd
import os

# --- SETUP PATHS ---
# This grabs the folder where this script is running
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'Robbery_Open_Data.csv')
output_path = os.path.join(base_dir, 'data', 'robbery_cleaned.csv')

print("--- Step 1: Preprocessing Started ---")

# 1. Load Data
if not os.path.exists(data_path):
    print(f"Error: File not found at {data_path}. Please make sure the CSV is in a 'data' subfolder.")
    exit()

df = pd.read_csv(data_path)
print(f"Raw data shape: {df.shape}") #

# 2. Data Cleaning (Logic from your original script)
# Remove columns that are not useful for prediction
cols_to_drop = ['Index_', 'event_unique_id', 'ObjectId']
# Check if columns exist before dropping to prevent errors
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

# Handle Missing Values
# Filling categorical text with 'Unknown' and numbers with 0 or median
df.fillna('Unknown', inplace=True) 

# 3. Feature Engineering
# Extract month/hour from Occurrence Date if needed, or just keep raw for now.
# (Assuming your group wants to keep it simple for the first version)

print(f"Cleaned data shape: {df.shape}")

# 4. Save to CSV for the next step
df.to_csv(output_path, index=False)
print(f"Processed data saved to: {output_path}")