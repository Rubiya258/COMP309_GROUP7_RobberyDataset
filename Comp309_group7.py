# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 16:00:29 2025

@author: RubiyaS
"""


import pandas as pd
import numpy as np

print("ROBBERY DATASET EXPLORATION")
print("=" * 50)

# 1. Load the dataset
print("\n1. Loading dataset...")
df = pd.read_csv('C:/Users/RubiyaS/Downloads/Robbery_Open_Data.csv')
print("Dataset loaded")

# 2. Display first few rows
print("\n2. First 5 rows:")
print(df.head())

# 3. Show column names
print("\n3. Column names:")
print(df.columns.tolist())

# 4. Show data types
print("\n4. Data types:")
print(df.dtypes)

# 5. Show number of rows & columns
print(f"\n5. Dataset shape: {df.shape}")
print(f"   Rows: {df.shape[0]}")
print(f"   Columns: {df.shape[1]}")

# 6. General description
print("\n6. Dataset info:")
df.info()

#------

# summary_statistics.py
import pandas as pd
import numpy as np

print("ROBBERY DATASET - SUMMARY STATISTICS")
print("=" * 50)


print("\n1. NUMERICAL STATISTICS")
print("-" * 30)

# Numerical columns statistics
numerical_cols = ['LONG_WGS84', 'LAT_WGS84', 'REPORT_HOUR', 'OCC_HOUR', 'UCR_EXT']
print(df[numerical_cols].describe())

print("\n2. CATEGORICAL STATISTICS")
print("-" * 30)



# UCR_EXT - Robbery subtypes
print("Robbery Subtypes (UCR_EXT):")
print(df['UCR_EXT'].value_counts())
most_common_robbery = df['UCR_EXT'].value_counts().index[0]
print(f"Most common robbery subtype: {most_common_robbery}")

# Neighbourhood analysis
print("\nNeighbourhood Analysis (NEIGHBOURHOOD_158):")
neighbourhood_counts = df['NEIGHBOURHOOD_158'].value_counts()
print(neighbourhood_counts.head(10))  # Top 10 neighbourhoods
highest_incident_area = neighbourhood_counts.index[0]
print(f"Neighbourhood with highest incidents: {highest_incident_area}")

# Premises type analysis
print("\nPremises Type Analysis:")
premises_counts = df['PREMISES_TYPE'].value_counts()
print(premises_counts)
most_common_premises = premises_counts.index[0]
print(f" Most common premises type: {most_common_premises}")

# Location type analysis
print("\nLocation Type Analysis:")
location_counts = df['LOCATION_TYPE'].value_counts()
print(location_counts)
most_common_location = location_counts.index[0]
print(f" Most common location type: {most_common_location}")

# Time analysis
print("\nTime of Day Analysis:")
print("Report Hours distribution:")
print(df['REPORT_HOUR'].value_counts().sort_index())
print("\nOccurrence Hours distribution:")
print(df['OCC_HOUR'].value_counts().sort_index())

# Find most common hour
most_common_hour = df['OCC_HOUR'].value_counts().index[0]
print(f" Most common occurrence hour: {most_common_hour}:00")

print("\n3. SUMMARY FOR REPORT")
print("-" * 30)
print(f"• Most common robbery subtype: {most_common_robbery}")
print(f"• Neighbourhood with highest incidents: {highest_incident_area}")
print(f"• Most common premises type: {most_common_premises}")
print(f"• Most common location type: {most_common_location}")
print(f"• Most common occurrence hour: {most_common_hour}:00")

#---------

print("\n1. MISSING VALUE COUNT FOR ALL COLUMNS")
print("-" * 40)

# Count missing values for all columns
missing_count = df.isnull().sum()
print("missing count: ", missing_count);

#NSA - Not Specified Area
nsa_columns = ['NEIGHBOURHOOD_158', 'NEIGHBOURHOOD_140']
for col in nsa_columns:
    if col in df.columns:
        nsa_count = (df[col] == 'NSA').sum()
        print(f"'{col}': {nsa_count} NSA values")
        
# Check for zeros in latitude and longitude
lat_zeros = (df['LAT_WGS84'] == 0).sum()
long_zeros = (df['LONG_WGS84'] == 0).sum()

print(f"LAT_WGS84 zeros: {lat_zeros} rows")
print(f"LONG_WGS84 zeros: {long_zeros} rows")

#--------------------------------------------


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

print("CREATING SEPARATE VISUALIZATIONS")
print("=" * 40)

# 1. Robbery subtype distribution (OFFENCE)
print("1. Creating Robbery Subtype Distribution...")
plt.figure(figsize=(10, 6))
robbery_types = df['OFFENCE'].value_counts()
robbery_types.plot(kind='bar')
plt.title('Robbery Subtype Distribution')
plt.xlabel('Robbery Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 2. Robbery count by hour
print("2. Creating Robbery Count by Hour...")
plt.figure(figsize=(10, 6))
sns.countplot(x='OCC_HOUR', data=df, order=sorted(df['OCC_HOUR'].unique()))
plt.title('Robbery Count by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


print("3. Creating Robbery Count by Month...")
plt.figure(figsize=(10, 6))

# Define the correct chronological order for months
month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

# Create the plot with proper month order
sns.countplot(x='OCC_MONTH', data=df, order=month_order)
plt.title('Robbery Count by Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# 4. Geographic scatter plot
print("4. Creating Geographic Distribution...")
plt.figure(figsize=(10, 8))
# Filter out zero coordinates for better visualization
geo_df = df[(df['LAT_WGS84'] != 0) & (df['LONG_WGS84'] != 0)]
plt.scatter(geo_df['LONG_WGS84'], geo_df['LAT_WGS84'], s=1, alpha=0.6)
plt.title('Geographic Distribution of Robberies')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.show()

