# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 21:43:42 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Prepare Binary Classification Dataset for Enzyme vs Non-Enzyme
Removes ID and metadata columns, converts labels to binary (1/0)
ADDS: Species labels for leave-one-species-out validation
"""

import pandas as pd
import numpy as np
import os

# =============================================
# Configuration
# =============================================
input_path = r"D:\zebfish\revision\zebfish_processed_results\combined_data\clean_fish_dataset_for_dl.csv"
output_path = r"D:\zebfish\revision\zebfish_processed_results\combined_data\binary_classification_dataset.csv"
species_output_path = r"D:\zebfish\revision\zebfish_processed_results\combined_data\binary_classification_with_species.csv"

print("=" * 60)
print("PREPARING BINARY CLASSIFICATION DATASET (WITH SPECIES)")
print("=" * 60)

# =============================================
# 1. Load the data
# =============================================
print(f"\n📂 Loading file: {input_path}")
df = pd.read_csv(input_path)
print(f"   Original shape: {df.shape}")
print(f"   Original columns: {df.columns.tolist()[:10]}...")

# =============================================
# 2. Remove unwanted columns (but KEEP Data_Source for species)
# =============================================
# Remove UniProt_ID and EC_Class, but KEEP Data_Source (species)
columns_to_remove = ['UniProt_ID', 'EC_Class']
columns_to_keep = [col for col in df.columns if col not in columns_to_remove]

df_clean = df[columns_to_keep].copy()
print(f"\n🗑️ Removed columns: {columns_to_remove}")
print(f"   Kept columns: {df_clean.columns.tolist()[:10]}...")
print(f"   New shape: {df_clean.shape}")

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
# 5. Check species distribution (Data_Source column)
# =============================================
print(f"\n🐟 Species distribution in dataset:")
species_dist = df_clean['Data_Source'].value_counts()
for species, count in species_dist.items():
    percentage = (count / len(df_clean)) * 100
    print(f"   {species:15s}: {count:>5} ({percentage:.1f}%)")

# =============================================
# 6. Save TWO versions:
#    a) Without species (for standard Code IV)
#    b) With species (for leave-one-species-out Code IV)
# =============================================

# Version A: Without species (standard classification)
df_no_species = df_clean.drop(columns=['Data_Source'])
df_no_species.to_csv(output_path, index=False)
print(f"\n💾 Saved standard dataset (no species) to: {output_path}")
print(f"   Shape: {df_no_species.shape}")

# Version B: With species (for leave-one-species-out)
df_with_species = df_clean.copy()
df_with_species.to_csv(species_output_path, index=False)
print(f"\n💾 Saved dataset with species to: {species_output_path}")
print(f"   Shape: {df_with_species.shape}")
print(f"   Columns: {df_with_species.columns.tolist()}")

# =============================================
# 7. Display sample of the data
# =============================================
print(f"\n📊 Sample of the dataset with species (first 5 rows):")
print(df_with_species[['Data_Source', 'Label'] + [f'Embedding_{i}' for i in range(3)]].head().to_string())

print(f"\n📊 Dataset info:")
print(f"   Total samples: {len(df_with_species)}")
print(f"   Features (embeddings): {df_with_species.shape[1] - 2}")  # Subtract Label and Data_Source
print(f"   Label column: 'Label' (1 = Enzyme, 0 = Non-enzyme)")
print(f"   Species column: 'Data_Source' (for leave-one-species-out CV)")

print("\n" + "=" * 60)
print("✅ DATASET PREPARATION COMPLETE!")
print("=" * 60)