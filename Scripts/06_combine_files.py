# -*- coding: utf-8 -*-
"""
Combine Results from Revision with Fold Data (FIXED - Correct LR Parsing)
Creates unified summary tables with fold-level values for statistical analysis
"""

import os
import pandas as pd
import numpy as np
import glob
import re

# =============================================
# Configuration
# =============================================

base_dir = r"D:\zebfish\revision\VARDON_Results_Corrected_0.01"
output_dir = r"D:\zebfish\revision\Combined_Results"
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("COMBINING REVISION RESULTS WITH FOLD DATA (CORRECT LR PARSING)")
print(f"Base directory: {base_dir}")
print(f"Output directory: {output_dir}")
print("=" * 70)

# =============================================
# 1. LOAD CV SUMMARY RESULTS
# =============================================

print("\n" + "=" * 70)
print("LOADING CV SUMMARY RESULTS")
print("=" * 70)

cv_file = os.path.join(base_dir, "Paper1_CV_Summary_Results.csv")
if os.path.exists(cv_file):
    combined_cv = pd.read_csv(cv_file)
    combined_cv['Learning_Rate'] = 0.01
    
    # Rename columns for consistency
    combined_cv = combined_cv.rename(columns={
        'learning_rate': 'LR',
        'batch_size': 'BS',
        'method': 'Method'
    })
    
    # Remove duplicate rows (keep first occurrence)
    combined_cv = combined_cv.drop_duplicates(subset=['Method', 'LR', 'BS'])
    
    print(f"✅ Loaded CV summary: {len(combined_cv)} rows")
    print(f"   Unique Methods: {combined_cv['Method'].nunique()}")
    print(f"   Unique LR: {sorted(combined_cv['LR'].unique())}")
    print(f"   Unique BS: {sorted(combined_cv['BS'].unique())}")
else:
    print(f"❌ Not found: {cv_file}")
    combined_cv = None

# =============================================
# 2. LOAD FOLD DATA - WITH CORRECT LR PARSING
# =============================================

print("\n" + "=" * 70)
print("LOADING FOLD-LEVEL DATA FROM Fold_Metrics.csv")
print("=" * 70)

# Find all Fold_Metrics.csv files
fold_files = glob.glob(os.path.join(base_dir, "cv_runs", "*", "*", "csv_files", "Fold_Metrics.csv"))
print(f"Found {len(fold_files)} Fold_Metrics.csv files")

fold_data_list = []

# Method names for path parsing
method_names = [
    'Logistic_Regression', 'Gaussian_Dropout_NN', 'Standard_Dropout', 
    'MLP_BatchNorm', 'Residual_MLP', 'VARDON_RealVD', 'VARDON_AdaptiveVD',
    'VARDON_RealVD_Adaptive', 'VARDON_Full', 'VARDON_Light',
    'VARDON_Full_No_Gate', 'VARDON_Full_No_Sparsity', 'VARDON_Full_No_Both'
]

def parse_lr_from_folder(folder_name):
    """
    Parse LR from folder name like 'lr_0_00010_bs_32' -> 0.0001
    """
    # Pattern: lr_0_00010_bs_32
    # or: lr_0_00100_bs_32
    # or: lr_0_01000_bs_32
    match = re.search(r'lr_0_(\d+)_bs_(\d+)', folder_name)
    if match:
        lr_str = match.group(1)  # "00010", "00100", "01000"
        # Remove trailing zeros and convert to float
        lr_str_clean = lr_str.rstrip('0')
        if lr_str_clean == "":
            lr_str_clean = "1"
        lr_val = float(f"0.{lr_str_clean}")
        return lr_val
    return None

for fold_file in fold_files:
    try:
        # Parse path: .../cv_runs/lr_0_00010_bs_32/Gaussian_Dropout_NN/csv_files/Fold_Metrics.csv
        path_parts = fold_file.split(os.sep)
        
        # Find method name
        method_name = None
        for i, part in enumerate(path_parts):
            if part in method_names:
                method_name = part
                break
        
        if method_name is None:
            # Try to find by pattern (method name is the folder before 'csv_files')
            for i, part in enumerate(path_parts):
                if part == 'csv_files' and i > 0:
                    method_name = path_parts[i-1]
                    if method_name in method_names:
                        break
        
        if method_name is None:
            continue
        
        # Find LR/BS folder
        lr_folder = None
        for part in path_parts:
            if 'lr_' in part and 'bs_' in part:
                lr_folder = part
                break
        
        if lr_folder is None:
            continue
        
        # Parse LR and BS
        lr_val = parse_lr_from_folder(lr_folder)
        if lr_val is None:
            continue
        
        # Parse BS
        bs_val = None
        match = re.search(r'bs_(\d+)', lr_folder)
        if match:
            bs_val = int(match.group(1))
        
        if bs_val is None:
            continue
        
        # Read fold metrics
        fold_df = pd.read_csv(fold_file)
        
        if len(fold_df) == 10:  # Should be 10 folds
            fold_dict = {
                'Method': method_name,
                'LR': lr_val,
                'BS': bs_val,
                'fold_accuracy_values': fold_df['Accuracy'].tolist(),
                'fold_precision_values': fold_df['Precision'].tolist(),
                'fold_recall_values': fold_df['Recall'].tolist(),
                'fold_f1_values': fold_df['F1'].tolist(),
                'fold_mcc_values': fold_df['MCC'].tolist(),
                'fold_auc_values': fold_df['AUC'].tolist(),
            }
            fold_data_list.append(fold_dict)
    except Exception as e:
        print(f"⚠️ Error loading {fold_file}: {e}")

# Convert to DataFrame and remove duplicates
fold_df = pd.DataFrame(fold_data_list)
fold_df = fold_df.drop_duplicates(subset=['Method', 'LR', 'BS'])
print(f"✅ Loaded fold data for {len(fold_df)} unique configurations")

# =============================================
# 3. DISPLAY SAMPLE FOLD DATA
# =============================================

print("\n" + "=" * 70)
print("SAMPLE FOLD DATA")
print("=" * 70)

if not fold_df.empty:
    print("\nLR values found in fold data:")
    print(sorted(fold_df['LR'].unique()))
    print(f"\nBS values found: {sorted(fold_df['BS'].unique())}")
    
    print("\nFirst 5 rows of fold data:")
    print(fold_df[['Method', 'LR', 'BS', 'fold_mcc_values']].head(5).to_string(index=False))
    
    print(f"\nTotal configurations with fold data: {len(fold_df)}")
    print(f"Expected: 13 methods × 3 LRs × 3 BS = 117")
    print(f"Found: {len(fold_df)}")

# =============================================
# 4. MERGE FOLD DATA WITH CV SUMMARY
# =============================================

print("\n" + "=" * 70)
print("MERGING FOLD DATA WITH CV SUMMARY")
print("=" * 70)

if combined_cv is not None and not fold_df.empty:
    # Ensure consistent column types
    combined_cv['LR'] = combined_cv['LR'].astype(float)
    combined_cv['BS'] = combined_cv['BS'].astype(int)
    
    fold_df['LR'] = fold_df['LR'].astype(float)
    fold_df['BS'] = fold_df['BS'].astype(int)
    
    # Print unique values to debug
    print(f"\nCV LR values: {sorted(combined_cv['LR'].unique())}")
    print(f"Fold LR values: {sorted(fold_df['LR'].unique())}")
    
    print(f"\nCV BS values: {sorted(combined_cv['BS'].unique())}")
    print(f"Fold BS values: {sorted(fold_df['BS'].unique())}")
    
    # Merge on Method, LR, BS
    combined_cv = pd.merge(
        combined_cv,
        fold_df,
        on=['Method', 'LR', 'BS'],
        how='left'
    )
    
    print(f"\n✅ Merged: {len(combined_cv)} rows")
    
    # Check fold columns
    fold_cols = [col for col in combined_cv.columns if col.startswith('fold_') and col.endswith('_values')]
    if fold_cols:
        print(f"✅ Fold columns present: {fold_cols}")
        
        # Check if any NaN values
        for col in fold_cols:
            nan_count = combined_cv[col].isna().sum()
            if nan_count > 0:
                print(f"   ⚠️ {col}: {nan_count} NaN values")
            else:
                print(f"   ✅ {col}: All values present")
        
        # Show sample with actual values
        print("\n📊 Sample fold data after merge:")
        sample = combined_cv[combined_cv['fold_mcc_values'].notna()][['Method', 'LR', 'BS', 'fold_mcc_values']].head(5)
        if not sample.empty:
            for idx, row in sample.iterrows():
                print(f"  {row['Method']}, LR={row['LR']}, BS={row['BS']}: {row['fold_mcc_values'][:3]}... (showing first 3 folds)")
        else:
            print("  ⚠️ No valid fold data found after merge")

# =============================================
# 5. SAVE COMBINED CV WITH FOLDS
# =============================================

if combined_cv is not None:
    cv_output_path = os.path.join(output_dir, "Combined_CV_Results_with_Folds.csv")
    combined_cv.to_csv(cv_output_path, index=False)
    print(f"\n✅ Combined CV results with folds saved to: {cv_output_path}")

# =============================================
# 6. LOAD TEST RESULTS
# =============================================

print("\n" + "=" * 70)
print("LOADING TEST RESULTS")
print("=" * 70)

test_file = os.path.join(base_dir, "Paper1_Test_Results_All_Models.csv")
if os.path.exists(test_file):
    combined_test = pd.read_csv(test_file)
    combined_test['Learning_Rate'] = 0.01
    test_output_path = os.path.join(output_dir, "Combined_Test_Results_All_Models.csv")
    combined_test.to_csv(test_output_path, index=False)
    print(f"✅ Loaded: {test_file} ({len(combined_test)} rows)")
    print(f"✅ Combined test results saved to: {test_output_path}")
else:
    print(f"❌ Not found: {test_file}")
    combined_test = None

# =============================================
# 7. LOAD SELECTED MODEL RESULTS
# =============================================

print("\n" + "=" * 70)
print("LOADING SELECTED MODEL RESULTS")
print("=" * 70)

selected_file = os.path.join(base_dir, "Paper1_Selected_Model_CV_and_Test_Result.csv")
if os.path.exists(selected_file):
    combined_selected = pd.read_csv(selected_file)
    combined_selected['Learning_Rate'] = 0.01
    selected_output_path = os.path.join(output_dir, "Combined_Selected_Model_Results.csv")
    combined_selected.to_csv(selected_output_path, index=False)
    print(f"✅ Loaded: {selected_file}")
    print(f"✅ Combined selected model results saved to: {selected_output_path}")
else:
    print(f"❌ Not found: {selected_file}")
    combined_selected = None

# =============================================
# 8. CREATE PAPER TABLE
# =============================================

print("\n" + "=" * 70)
print("CREATING PAPER TABLE: BEST MODEL PER CONFIGURATION")
print("=" * 70)

if combined_test is not None and not combined_test.empty:
    paper_table = combined_test.groupby(['Method', 'Learning_Rate']).agg({
        'Batch_Size': lambda x: x.iloc[0],
        'Test_Accuracy': 'max',
        'Test_Precision': 'max',
        'Test_Recall': 'max',
        'Test_F1': 'max',
        'Test_MCC': 'max',
        'Test_AUC': 'max',
        'CV_Mean_Accuracy': 'first',
        'CV_Mean_MCC': 'first',
        'Feature_Stability_Jaccard': 'first'
    }).reset_index()
    
    paper_table = paper_table.sort_values('Test_MCC', ascending=False)
    paper_table_path = os.path.join(output_dir, "Paper_Best_Results_Table.csv")
    paper_table.to_csv(paper_table_path, index=False)
    
    print("\n📊 BEST RESULTS FOR PAPER:")
    print("=" * 70)
    print(paper_table[['Method', 'Learning_Rate', 'Batch_Size', 'Test_Accuracy',
                       'Test_F1', 'Test_MCC', 'Test_AUC', 'Feature_Stability_Jaccard']].to_string(index=False))
    print(f"\n✅ Paper table saved to: {paper_table_path}")

# =============================================
# 9. SUMMARY
# =============================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

if combined_cv is not None and not combined_cv.empty:
    print("\n🏆 Best CV Performance by Method:")
    best_cv = combined_cv.loc[combined_cv.groupby('Method')['mean_mcc'].idxmax()]
    best_cv = best_cv.sort_values('mean_mcc', ascending=False)
    for _, row in best_cv.iterrows():
        print(f"  {row['Method']:<30}: MCC={row['mean_mcc']:.4f} (BS={row['BS']})")

if combined_test is not None and not combined_test.empty:
    print("\n🏆 Best Test Performance by Method:")
    best_test = combined_test.loc[combined_test.groupby('Method')['Test_MCC'].idxmax()]
    best_test = best_test.sort_values('Test_MCC', ascending=False)
    for _, row in best_test.iterrows():
        print(f"  {row['Method']:<30}: Test MCC={row['Test_MCC']:.4f} (BS={row['Batch_Size']})")

# =============================================
# 10. OVERALL BEST MODEL
# =============================================

print("\n" + "=" * 70)
print("🏆 OVERALL BEST MODEL")
print("=" * 70)

if combined_selected is not None and not combined_selected.empty:
    best_overall = combined_selected.iloc[0]
    print(f"Method:              {best_overall['Selected_Method']}")
    print(f"Description:         {best_overall['Description']}")
    print(f"Learning Rate:       0.01")
    print(f"Batch Size:          {best_overall['Batch_Size']}")
    print(f"CV Mean MCC:         {best_overall['CV_Mean_MCC']:.4f} ± {best_overall['CV_Std_MCC']:.4f}")
    print(f"Test Accuracy:       {best_overall['Test_Accuracy']:.4f}")
    print(f"Test F1:             {best_overall['Test_F1']:.4f}")
    print(f"Test MCC:            {best_overall['Test_MCC']:.4f}")
    print(f"Test AUC:            {best_overall['Test_AUC']:.4f}")
    print(f"Feature Stability:   {best_overall['Feature_Stability_Jaccard']:.4f}")

print("\n" + "=" * 70)
print("✅ ALL RESULTS COMBINED!")
print(f"📁 Combined results saved to: {output_dir}")
print("=" * 70)