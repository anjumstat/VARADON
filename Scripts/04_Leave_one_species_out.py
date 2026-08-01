# -*- coding: utf-8 -*-
"""
03_create_clean_splits_fish.py
Create clean CSV files with ONLY embeddings and target variable for FISH dataset
No UniProt_ID, no species, no EC_Class - just features and labels
Designed for leave-one-species-out evaluation
"""

import pandas as pd
import os
import numpy as np

# =============================================
# CONFIGURATION
# =============================================

class Config:
    # Path to the binary classification dataset with species (from Code III)
    INPUT_FILE = r"D:\zebfish\revision\zebfish_processed_results\combined_data\binary_classification_with_species.csv"
    
    # Output directory for clean splits
    OUTPUT_DIR = r"D:\zebfish\revision\data\clean_splits"
    
    EMBEDDING_DIM = 1024

# =============================================
# LOAD DATA WITH SPECIES
# =============================================

def load_data_with_species(config):
    """
    Load the dataset that has species information
    """
    print("=" * 80)
    print("LOADING FISH DATASET WITH SPECIES")
    print("=" * 80)
    
    if not os.path.exists(config.INPUT_FILE):
        print(f"❌ Input file not found: {config.INPUT_FILE}")
        return None
    
    df = pd.read_csv(config.INPUT_FILE)
    print(f"✅ Loaded {len(df)} samples")
    print(f"   Columns: {df.columns.tolist()[:5]}... (plus embedding columns)")
    print(f"   Species: {df['Data_Source'].unique()}")
    print(f"   Enzymes (1): {df['Label'].sum()}")
    print(f"   Non-enzymes (0): {len(df) - df['Label'].sum()}")
    
    return df

# =============================================
# CREATE CLEAN SPLITS FOR LOSO
# =============================================

def create_loso_clean_splits(df, config):
    """
    Create clean splits for each species (leave-one-species-out)
    """
    print("\n" + "=" * 80)
    print("CREATING CLEAN LOSO SPLITS FOR FISH DATASET")
    print("=" * 80)
    
    # Get unique species
    species_list = df['Data_Source'].unique()
    print(f"\n🐟 Species found: {species_list}")
    
    # Create output directories
    loso_output_dir = os.path.join(config.OUTPUT_DIR, 'loso')
    os.makedirs(loso_output_dir, exist_ok=True)
    
    # Get embedding column names
    emb_cols = [f'Embedding_{i}' for i in range(config.EMBEDDING_DIM)]
    
    # Check if embedding columns exist
    available_cols = [col for col in emb_cols if col in df.columns]
    if len(available_cols) != config.EMBEDDING_DIM:
        print(f"   ⚠️ Found {len(available_cols)}/{config.EMBEDDING_DIM} embedding columns")
    
    print(f"\n📊 Creating clean splits for {len(species_list)} species...")
    print("-" * 80)
    
    species_summary = []
    
    for species in species_list:
        print(f"\n📁 Species: {species}")
        
        # Split data by species
        species_df = df[df['Data_Source'] == species]
        other_df = df[df['Data_Source'] != species]
        
        # Create clean dataframes (embeddings + label only)
        def make_clean(df_subset, name):
            if len(df_subset) == 0:
                return None
            
            clean_df = df_subset[available_cols + ['Label']].copy()
            
            # Convert embeddings to float32
            for col in available_cols:
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').astype(np.float32)
            
            # Ensure label is int
            clean_df['Label'] = clean_df['Label'].astype(np.int8)
            
            # Calculate statistics
            total = len(clean_df)
            enzymes = clean_df['Label'].sum()
            non_enzymes = total - enzymes
            
            print(f"   {name}: {total} samples (Enzymes: {enzymes}, Non-enzymes: {non_enzymes})")
            
            return clean_df
        
        # Create clean train and test
        clean_train = make_clean(other_df, "Train")
        clean_test = make_clean(species_df, "Test")
        
        # Save if both exist
        if clean_train is not None and clean_test is not None:
            # Create species directory
            species_dir = os.path.join(loso_output_dir, species)
            os.makedirs(species_dir, exist_ok=True)
            
            # Save files
            train_path = os.path.join(species_dir, 'train.csv')
            test_path = os.path.join(species_dir, 'test.csv')
            
            clean_train.to_csv(train_path, index=False)
            clean_test.to_csv(test_path, index=False)
            
            print(f"   ✅ Saved to: {species_dir}/")
            
            # Store summary
            species_summary.append({
                'Species': species,
                'Train_Size': len(clean_train),
                'Test_Size': len(clean_test),
                'Train_Enzymes': int(clean_train['Label'].sum()),
                'Test_Enzymes': int(clean_test['Label'].sum()),
                'Train_Non_Enzymes': len(clean_train) - int(clean_train['Label'].sum()),
                'Test_Non_Enzymes': len(clean_test) - int(clean_test['Label'].sum())
            })
        else:
            print(f"   ❌ Skipping {species} - missing data")
    
    return pd.DataFrame(species_summary)

# =============================================
# CREATE COMBINED CLEAN DATASET
# =============================================

def create_combined_clean(df, config):
    """
    Create a single clean dataset with all data (for reference)
    """
    print("\n" + "=" * 80)
    print("CREATING COMBINED CLEAN DATASET")
    print("=" * 80)
    
    # Get embedding column names
    emb_cols = [f'Embedding_{i}' for i in range(config.EMBEDDING_DIM)]
    available_cols = [col for col in emb_cols if col in df.columns]
    
    # Create clean dataframe (only embeddings + label)
    clean_df = df[available_cols + ['Label']].copy()
    
    # Convert types
    for col in available_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').astype(np.float32)
    
    clean_df['Label'] = clean_df['Label'].astype(np.int8)
    
    # Statistics
    total = len(clean_df)
    enzymes = clean_df['Label'].sum()
    non_enzymes = total - enzymes
    
    print(f"   Combined clean dataset:")
    print(f"   Total samples: {total}")
    print(f"   Enzymes: {enzymes}")
    print(f"   Non-enzymes: {non_enzymes}")
    print(f"   Shape: {clean_df.shape}")
    
    # Save
    output_file = os.path.join(config.OUTPUT_DIR, 'combined_clean.csv')
    clean_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to: {output_file}")
    
    return clean_df

# =============================================
# CREATE TRAIN/VAL/TEST SPLITS FOR STANDARD CV
# =============================================

def create_standard_splits(df, config, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    Create standard train/val/test splits (optional, for reference)
    """
    print("\n" + "=" * 80)
    print("CREATING STANDARD TRAIN/VAL/TEST SPLITS")
    print("=" * 80)
    
    from sklearn.model_selection import train_test_split
    
    # Get embedding columns
    emb_cols = [f'Embedding_{i}' for i in range(config.EMBEDDING_DIM)]
    available_cols = [col for col in emb_cols if col in df.columns]
    
    # Create feature matrix and labels
    X = df[available_cols].values.astype(np.float32)
    y = df['Label'].values.astype(np.int8)
    
    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        random_state=42,
        stratify=y
    )
    
    # Second split: train vs val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio_adjusted,
        random_state=42,
        stratify=y_train_val
    )
    
    # Create dataframes
    train_df = pd.DataFrame(X_train, columns=available_cols)
    train_df['Label'] = y_train
    
    val_df = pd.DataFrame(X_val, columns=available_cols)
    val_df['Label'] = y_val
    
    test_df = pd.DataFrame(X_test, columns=available_cols)
    test_df['Label'] = y_test
    
    # Save
    standard_dir = os.path.join(config.OUTPUT_DIR, 'standard_splits')
    os.makedirs(standard_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(standard_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(standard_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(standard_dir, 'test.csv'), index=False)
    
    print(f"\n✅ Standard splits saved to: {standard_dir}/")
    print(f"   Train: {len(train_df)} samples (Enzymes: {y_train.sum()})")
    print(f"   Val:   {len(val_df)} samples (Enzymes: {y_val.sum()})")
    print(f"   Test:  {len(test_df)} samples (Enzymes: {y_test.sum()})")
    
    return train_df, val_df, test_df

# =============================================
# MAIN EXECUTION
# =============================================

def main():
    print("=" * 80)
    print("CREATE CLEAN SPLITS FOR FISH DATASET")
    print("Leave-One-Species-Out Evaluation")
    print("=" * 80)
    
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # =============================================
    # 1. Load Data
    # =============================================
    df = load_data_with_species(config)
    if df is None:
        return
    
    # =============================================
    # 2. Create LOSO Clean Splits (MAIN OUTPUT)
    # =============================================
    loso_summary = create_loso_clean_splits(df, config)
    
    # Save LOSO summary
    loso_summary_path = os.path.join(config.OUTPUT_DIR, 'loso_summary.csv')
    loso_summary.to_csv(loso_summary_path, index=False)
    print(f"\n✅ LOSO summary saved to: {loso_summary_path}")
    
    # =============================================
    # 3. Create Combined Clean Dataset (Optional)
    # =============================================
    combined_df = create_combined_clean(df, config)
    
    # =============================================
    # 4. Create Standard Splits (Optional)
    # =============================================
    train_df, val_df, test_df = create_standard_splits(df, config)
    
    # =============================================
    # 5. Final Summary
    # =============================================
    print("\n" + "=" * 80)
    print("✅ CLEAN SPLITS CREATION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Output directory: {config.OUTPUT_DIR}")
    
    print("\n📊 LOSO Split Summary:")
    print("-" * 80)
    print(loso_summary.to_string(index=False))
    
    print("\n📁 Directory Structure:")
    print(f"  {config.OUTPUT_DIR}/")
    print(f"  ├── combined_clean.csv  (All data: embeddings + Label)")
    print(f"  ├── loso_summary.csv    (Summary of LOSO splits)")
    print(f"  ├── standard_splits/    (Standard train/val/test splits)")
    print(f"  │   ├── train.csv")
    print(f"  │   ├── val.csv")
    print(f"  │   └── test.csv")
    print(f"  └── loso/               (Leave-one-species-out splits)")
    
    for species in df['Data_Source'].unique():
        print(f"      ├── {species}/")
        print(f"      │   ├── train.csv  (All other species)")
        print(f"      │   └── test.csv   (Only {species})")
    
    print("\n" + "=" * 80)
    print("Each CSV file contains ONLY:")
    print("  - 1024 embedding columns (Embedding_0 to Embedding_1023)")
    print("  - 1 label column (Label: 1=Enzyme, 0=Non-enzyme)")
    print("=" * 80)

if __name__ == "__main__":
    main()