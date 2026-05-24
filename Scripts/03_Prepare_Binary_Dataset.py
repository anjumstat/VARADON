# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:03:00 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Prepare Binary Classification Dataset for Enzyme vs Non-Enzyme
Removes ID and metadata columns, converts labels to binary (1/0)
"""

import pandas as pd
import numpy as np
import os

# =============================================
# Configuration
# =============================================
input_path = r"D:\zebfish_processed_results\combined_data\clean_fish_dataset_for_dl.csv"
output_path = r"D:\zebfish_processed_results\combined_data\binary_classification_dataset.csv"

print("=" * 60)
print("PREPARING BINARY CLASSIFICATION DATASET")
print("=" * 60)

# =============================================
# 1. Load the data
# =============================================
print(f"\n📂 Loading file: {input_path}")
df = pd.read_csv(input_path)
print(f"   Original shape: {df.shape}")
print(f"   Original columns: {df.columns.tolist()}")

# =============================================
# 2. Remove unwanted columns
# =============================================
columns_to_remove = ['UniProt_ID', 'EC_Class', 'Data_Source']
columns_to_keep = [col for col in df.columns if col not in columns_to_remove]

df_clean = df[columns_to_keep].copy()
print(f"\n🗑️ Removed columns: {columns_to_remove}")
print(f"   New shape: {df_clean.shape}")
print(f"   Remaining columns: {df_clean.columns.tolist()}")

# =============================================
# 3. Convert labels to binary (1 = Enzyme, 0 = Non-enzyme)
# =============================================
print(f"\n🏷️ Converting labels to binary...")
print(f"   Original label distribution:")
print(f"     {df_clean['Enzyme_Classification'].value_counts().to_dict()}")

# Convert: Enzyme -> 1, Non-enzyme -> 0
df_clean['Enzyme_Classification'] = df_clean['Enzyme_Classification'].map({
    'Enzyme': 1,
    'Non-enzyme': 0
})

print(f"\n   New label distribution:")
print(f"     Enzyme (1): {(df_clean['Enzyme_Classification'] == 1).sum()}")
print(f"     Non-enzyme (0): {(df_clean['Enzyme_Classification'] == 0).sum()}")

# =============================================
# 4. Rename the label column for clarity
# =============================================
df_clean.rename(columns={'Enzyme_Classification': 'Label'}, inplace=True)
print(f"\n📝 Renamed 'Enzyme_Classification' to 'Label'")

# =============================================
# 5. Verify data types
# =============================================
print(f"\n🔍 Data type check:")
print(f"   Label column type: {df_clean['Label'].dtype}")
print(f"   Embedding columns type: {df_clean.iloc[:, 1:].dtypes.iloc[0]}")

# =============================================
# 6. Save the cleaned dataset
# =============================================
df_clean.to_csv(output_path, index=False)
print(f"\n💾 Saved to: {output_path}")
print(f"   Final shape: {df_clean.shape}")

# =============================================
# 7. Display sample of the data
# =============================================
print(f"\n📊 Sample of the cleaned dataset (first 5 rows, first 5 columns):")
print(df_clean.iloc[:5, :5].to_string())

print(f"\n📊 Dataset info:")
print(f"   Total samples: {len(df_clean)}")
print(f"   Features (embeddings): {df_clean.shape[1] - 1}")  # Subtract label column
print(f"   Label column: 'Label' (1 = Enzyme, 0 = Non-enzyme)")

print("\n" + "=" * 60)
print("✅ DATASET PREPARATION COMPLETE!")
print("=" * 60)