# -*- coding: utf-8 -*-
"""
Combine all Fold_Metrics.csv files from VARDON experiments
CORRECTED: Properly parses learning rates from folder names
"""

import os
import pandas as pd
import glob
import re

# =============================================
# CONFIGURATION
# =============================================

base_dirs = [
    r"D:\zebfish\VARDON\VARDON_Results_Corrected_0.01",
    r"D:\zebfish\VARDON\VARDON_Results_Corrected_0.001",
    r"D:\zebfish\VARDON\VARDON_Results_Corrected_0.0001",
]

output_dir = r"D:\zebfish\VARDON\Combined_Results\CV_all_models"
os.makedirs(output_dir, exist_ok=True)

# Methods in order
methods = [
    'Logistic_Regression',
    'Gaussian_Dropout_NN',
    'Standard_Dropout',
    'VARDON_RealVD',
    'VARDON_AdaptiveVD',
    'VARDON_RealVD_Adaptive',
    'VARDON_Full',
    'VARDON_Light'
]

def parse_lr_bs(folder_name):
    """
    Parse learning rate and batch size from folder name
    Examples:
        'lr_0_01000_bs_32' -> (0.01, 32)
        'lr_0_00100_bs_64' -> (0.001, 64)
        'lr_0_00010_bs_128' -> (0.0001, 128)
    """
    # Pattern: lr_0_01000_bs_32
    match = re.search(r'lr_0_(\d+)_bs_(\d+)', folder_name)
    if match:
        lr_str = match.group(1)  # "01000", "00100", "00010"
        # Convert "01000" -> 0.01
        # Remove trailing zeros and divide
        lr_str_clean = lr_str.rstrip('0')
        if lr_str_clean == "":
            lr_str_clean = "1"
        lr = float(f"0.{lr_str_clean}")
        bs = int(match.group(2))
        return lr, bs
    return None, None

print("=" * 80)
print("COMBINING ALL FOLD_METRICS.CSV FILES (CORRECTED)")
print("=" * 80)

# =============================================
# 1. Collect all Fold_Metrics.csv files
# =============================================

all_fold_metrics = []

for base_dir in base_dirs:
    print(f"\n📁 Scanning: {base_dir}")
    
    # Find all Fold_Metrics.csv files
    pattern = os.path.join(base_dir, "cv_runs", "*", "*", "csv_files", "Fold_Metrics.csv")
    files = glob.glob(pattern)
    
    print(f"   Found {len(files)} Fold_Metrics.csv files")
    
    for file_path in files:
        # Extract information from path
        # Example path: .../cv_runs/lr_0_01000_bs_32/Gaussian_Dropout_NN/csv_files/Fold_Metrics.csv
        path_parts = file_path.split(os.sep)
        
        # Find the learning rate/batch size folder and method name
        lr_bs_folder = None
        method_name = None
        
        for i, part in enumerate(path_parts):
            if 'lr_' in part and 'bs_' in part:
                lr_bs_folder = part
            if part in methods:
                method_name = part
        
        if lr_bs_folder and method_name:
            # Parse learning rate and batch size
            lr, bs = parse_lr_bs(lr_bs_folder)
            
            if lr is not None:
                # Read the CSV
                df = pd.read_csv(file_path)
                
                # Add metadata columns
                df['Method_Name'] = method_name
                df['Learning_Rate'] = lr
                df['Batch_Size'] = bs
                
                # Add configuration identifier
                df['Config'] = f"lr_{lr}_bs_{bs}"
                
                all_fold_metrics.append(df)
                
                print(f"   ✅ {method_name} | LR={lr} | BS={bs} | {len(df)} folds")
            else:
                print(f"   ⚠️ Could not parse: {lr_bs_folder}")
        else:
            print(f"   ⚠️ Skipping: {os.path.basename(file_path)}")

print(f"\n📊 Total files collected: {len(all_fold_metrics)}")

# =============================================
# 2. Combine all data
# =============================================

if all_fold_metrics:
    combined_df = pd.concat(all_fold_metrics, ignore_index=True)
    
    print(f"\n📊 Combined DataFrame shape: {combined_df.shape}")
    print(f"   Columns: {combined_df.columns.tolist()}")
    
    # Reorder columns for better readability
    column_order = ['Method_Name', 'Learning_Rate', 'Batch_Size', 'Fold', 
                    'Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'AUC',
                    'Training_Time_Seconds', 'Epochs_Trained', 'Config']
    
    # Keep only columns that exist
    column_order = [col for col in column_order if col in combined_df.columns]
    combined_df = combined_df[column_order]
    
    # Sort by Method, Learning Rate, Batch Size, Fold
    combined_df = combined_df.sort_values(['Method_Name', 'Learning_Rate', 'Batch_Size', 'Fold'])
    
    # Save combined file
    output_path = os.path.join(output_dir, "ALL_METHODS_Fold_Metrics_Combined.csv")
    combined_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Combined file saved to: {output_path}")
    
    # Display summary
    print("\n📊 SUMMARY OF COMBINED DATA:")
    print(f"   Total rows: {len(combined_df)}")
    print(f"   Methods: {sorted(combined_df['Method_Name'].unique())}")
    print(f"   Learning Rates: {sorted(combined_df['Learning_Rate'].unique())}")
    print(f"   Batch Sizes: {sorted(combined_df['Batch_Size'].unique())}")
    
    # Verify folds per configuration
    folds_per_config = combined_df.groupby(['Method_Name', 'Learning_Rate', 'Batch_Size']).size().iloc[0] if not combined_df.empty else 0
    print(f"   Folds per configuration: {folds_per_config}")
    
    # =============================================
    # 3. Per-method summary
    # =============================================
    
    print("\n📊 PER-METHOD SUMMARY (Mean ± Std across all configurations):")
    method_summary = combined_df.groupby('Method_Name').agg({
        'Accuracy': ['mean', 'std'],
        'Precision': ['mean', 'std'],
        'Recall': ['mean', 'std'],
        'F1': ['mean', 'std'],
        'MCC': ['mean', 'std'],
        'AUC': ['mean', 'std']
    }).round(4)
    
    print(method_summary)
    
    # Save method summary
    method_summary_path = os.path.join(output_dir, "Method_Summary_Statistics.csv")
    method_summary.to_csv(method_summary_path)
    
    # =============================================
    # 4. Best configuration per method (by MCC)
    # =============================================
    
    best_configs = combined_df.loc[combined_df.groupby('Method_Name')['MCC'].idxmax()]
    best_configs = best_configs[['Method_Name', 'Learning_Rate', 'Batch_Size', 'MCC', 'Accuracy', 'F1', 'AUC']]
    best_configs = best_configs.sort_values('MCC', ascending=False)
    
    best_configs_path = os.path.join(output_dir, "Best_Configuration_Per_Method.csv")
    best_configs.to_csv(best_configs_path, index=False)
    
    print("\n🏆 BEST CONFIGURATION PER METHOD (by MCC):")
    print(best_configs.to_string(index=False))
    
    # =============================================
    # 5. Per Learning Rate Summary
    # =============================================
    
    print("\n📊 PERFORMANCE BY LEARNING RATE (Mean MCC across methods):")
    lr_summary = combined_df.groupby('Learning_Rate').agg({
        'MCC': ['mean', 'std'],
        'Accuracy': ['mean', 'std'],
        'F1': ['mean', 'std']
    }).round(4)
    print(lr_summary)
    
    lr_summary_path = os.path.join(output_dir, "By_Learning_Rate_Summary.csv")
    lr_summary.to_csv(lr_summary_path)
    
else:
    print("❌ No files found! Please check the directory structure.")

print("\n" + "=" * 80)
print("✅ COMBINATION COMPLETE!")
print(f"📁 Output directory: {output_dir}")
print("=" * 80)