# -*- coding: utf-8 -*-
"""
Generate REAL ROC Curves from actual predictions and true labels
UPDATED: For revision results - all results in one location with folder structure:
D:\zebfish\revision\VARDON_Results_Corrected_0.01\cv_runs\lr_0_00010_bs_32\method\
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import glob
import re

# =============================================
# CONFIGURATION
# =============================================

# SINGLE base directory (all results are here)
BASE_DIR = r"D:\zebfish\revision\VARDON_Results_Corrected_0.01"

# All 13 methods
methods = [
    'Logistic_Regression',
    'Gaussian_Dropout_NN',
    'Standard_Dropout',
    'MLP_BatchNorm',
    'Residual_MLP',
    'VARDON_RealVD',
    'VARDON_AdaptiveVD',
    'VARDON_RealVD_Adaptive',
    'VARDON_Full',
    'VARDON_Light',
    'VARDON_Full_No_Gate',
    'VARDON_Full_No_Sparsity',
    'VARDON_Full_No_Both'
]

# Best configurations from your Table 3 (Revision results)
best_configs = {
    'Logistic_Regression': {'lr': 0.01, 'bs': 64},
    'Gaussian_Dropout_NN': {'lr': 0.01, 'bs': 32},
    'Standard_Dropout': {'lr': 0.01, 'bs': 128},
    'MLP_BatchNorm': {'lr': 0.01, 'bs': 64},
    'Residual_MLP': {'lr': 0.01, 'bs': 32},
    'VARDON_RealVD': {'lr': 0.01, 'bs': 32},
    'VARDON_AdaptiveVD': {'lr': 0.01, 'bs': 128},
    'VARDON_RealVD_Adaptive': {'lr': 0.01, 'bs': 32},
    'VARDON_Full': {'lr': 0.01, 'bs': 32},
    'VARDON_Light': {'lr': 0.01, 'bs': 64},
    'VARDON_Full_No_Gate': {'lr': 0.01, 'bs': 32},
    'VARDON_Full_No_Sparsity': {'lr': 0.01, 'bs': 32},
    'VARDON_Full_No_Both': {'lr': 0.01, 'bs': 64}
}

# Colors for all methods
method_colors = {
    # Baselines
    'Logistic_Regression': '#808080',           # Gray
    'Gaussian_Dropout_NN': '#1f77b4',           # Blue
    'Standard_Dropout': '#ff7f0e',              # Orange
    'MLP_BatchNorm': '#2ca02c',                 # Green
    'Residual_MLP': '#17becf',                  # Cyan
    
    # VARDON Variants
    'VARDON_RealVD': '#9467bd',                 # Purple
    'VARDON_AdaptiveVD': '#d62728',             # Red
    'VARDON_RealVD_Adaptive': '#e377c2',        # Pink
    'VARDON_Full': '#8c564b',                   # Brown
    'VARDON_Light': '#bcbd22',                  # Olive
    
    # VARDON Ablations
    'VARDON_Full_No_Gate': '#7f7f7f',           # Dark Gray
    'VARDON_Full_No_Sparsity': '#f7b6d2',       # Light Pink
    'VARDON_Full_No_Both': '#98df8a',           # Light Green
}

# Method groups for legend
method_groups = {
    'Baselines': ['Logistic_Regression', 'Gaussian_Dropout_NN', 'Standard_Dropout', 
                  'MLP_BatchNorm', 'Residual_MLP'],
    'VARDON Variants': ['VARDON_RealVD', 'VARDON_AdaptiveVD', 'VARDON_RealVD_Adaptive', 
                        'VARDON_Full', 'VARDON_Light'],
    'VARDON Ablations': ['VARDON_Full_No_Gate', 'VARDON_Full_No_Sparsity', 'VARDON_Full_No_Both']
}


def get_lr_folder_name(lr, bs):
    """
    Convert LR and BS to folder name format
    Examples:
        lr=0.01, bs=32 -> lr_0_01000_bs_32
        lr=0.001, bs=64 -> lr_0_00100_bs_64
        lr=0.0001, bs=128 -> lr_0_00010_bs_128
    """
    # Convert LR to folder format
    lr_str = f"{lr:.4f}"  # "0.0100", "0.0010", "0.0001"
    lr_parts = lr_str.split('.')
    if len(lr_parts) == 2:
        # Pad to 5 digits after decimal
        frac = lr_parts[1].ljust(5, '0')[:5]
        lr_folder = f"lr_{lr_parts[0]}_{frac}"
    else:
        lr_folder = f"lr_0_{int(lr*100000):05d}"
    
    return f"{lr_folder}_bs_{bs}"


def load_roc_data(method, lr, bs):
    """Load predictions and true labels from all folds to compute ROC"""
    
    all_probs = []
    all_labels = []
    
    # Get the folder name for this LR/BS
    lr_folder = get_lr_folder_name(lr, bs)
    
    # Build path: BASE_DIR/cv_runs/lr_0_01000_bs_32/method/npy_files/
    method_dir = os.path.join(BASE_DIR, "cv_runs", lr_folder, method)
    npy_dir = os.path.join(method_dir, "npy_files")
    
    if not os.path.exists(npy_dir):
        print(f"    ⚠️ Directory not found: {npy_dir}")
        return None, None, None
    
    # Load all 10 folds
    for fold in range(1, 11):
        pred_file = os.path.join(npy_dir, f"fold{fold}_predictions.npy")
        labels_file = os.path.join(npy_dir, f"fold{fold}_true_labels.npy")
        
        if os.path.exists(pred_file) and os.path.exists(labels_file):
            try:
                pred = np.load(pred_file)
                labels = np.load(labels_file)
                
                # Get probability for class 1 (Enzyme)
                if pred.shape[1] == 2:
                    probs = pred[:, 1]
                else:
                    probs = pred
                
                all_probs.extend(probs)
                all_labels.extend(labels)
            except Exception as e:
                print(f"    ⚠️ Error loading fold {fold}: {e}")
    
    if len(all_probs) > 0:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc
    
    return None, None, None


# =============================================
# GENERATE REAL ROC CURVES
# =============================================

print("\n" + "="*60)
print("Generating REAL ROC Curves from Actual Data (Revision)")
print(f"Base directory: {BASE_DIR}")
print("="*60)

fig, ax = plt.subplots(figsize=(12, 10))

# Store AUC values for sorting
roc_results = []

for method in methods:
    config = best_configs[method]
    lr = config['lr']
    bs = config['bs']
    
    print(f"\n📊 Processing: {method} (LR={lr}, BS={bs})")
    fpr, tpr, roc_auc = load_roc_data(method, lr, bs)
    
    if fpr is not None:
        roc_results.append({
            'method': method,
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr
        })
        print(f"    ✅ AUC = {roc_auc:.4f}")
    else:
        print(f"    ❌ No ROC data found")

# Sort by AUC (descending)
roc_results.sort(key=lambda x: x['auc'], reverse=True)

# Plot each method
for result in roc_results:
    method = result['method']
    roc_auc = result['auc']
    fpr = result['fpr']
    tpr = result['tpr']
    
    # Determine line style
    if method in method_groups['Baselines']:
        linestyle = '--'
        linewidth = 2
        alpha = 0.9
    elif method in method_groups['VARDON Ablations']:
        linestyle = ':'
        linewidth = 2
        alpha = 0.8
    else:  # VARDON Variants
        linestyle = '-'
        linewidth = 2.5
        alpha = 0.95
    
    # Plot
    ax.plot(fpr, tpr, linewidth=linewidth, linestyle=linestyle,
            color=method_colors.get(method, '#1f77b4'),
            alpha=alpha,
            label=f'{method} (AUC = {roc_auc:.4f})')

# Diagonal line (random classifier)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.5)')

# Labels and title
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14)
ax.set_title('Figure 2. Receiver Operating Characteristic (ROC) Curves\nEnzyme vs Non-enzyme Classification', 
             fontsize=16, fontweight='bold')
ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

# Add annotation for top performers
if roc_results:
    best_auc = roc_results[0]
    ax.annotate(f'Best: {best_auc["method"]}\nAUC = {best_auc["auc"]:.4f}',
                xy=(0.05, 0.05), xycoords='axes fraction',
                fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))

plt.tight_layout()

# =============================================
# SAVE FIGURES
# =============================================

output_dir = r"D:\zebfish\revision\Paper_Figures"
png_dir = os.path.join(output_dir, "PNG")
tiff_dir = os.path.join(output_dir, "TIFF")
svg_dir = os.path.join(output_dir, "SVG")

for d in [png_dir, tiff_dir, svg_dir]:
    os.makedirs(d, exist_ok=True)

# Save formats
png_path = os.path.join(png_dir, "Figure2_ROC_Curves_Revision.png")
tiff_path = os.path.join(tiff_dir, "Figure2_ROC_Curves_Revision.tiff")
svg_path = os.path.join(svg_dir, "Figure2_ROC_Curves_Revision.svg")
pdf_path = os.path.join(output_dir, "Figure2_ROC_Curves_Revision.pdf")

fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(tiff_path, dpi=300, bbox_inches='tight', facecolor='white', 
            format='tiff', pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')

print(f"\n✅ Saved PNG: {png_path}")
print(f"✅ Saved TIFF: {tiff_path}")
print(f"✅ Saved SVG: {svg_path}")
print(f"✅ Saved PDF: {pdf_path}")

# =============================================
# PRINT SUMMARY
# =============================================

print("\n" + "="*60)
print("ROC CURVES SUMMARY")
print("="*60)

# Top 5 performers
print("\n🏆 TOP 5 PERFORMERS (by AUC):")
for i, result in enumerate(roc_results[:5], 1):
    print(f"   {i}. {result['method']}: {result['auc']:.4f}")

# Group by category
print("\n📊 Performance by Category:")
for group_name, methods_list in method_groups.items():
    group_aucs = [r['auc'] for r in roc_results if r['method'] in methods_list]
    if group_aucs:
        print(f"   {group_name}:")
        print(f"      Mean AUC: {np.mean(group_aucs):.4f} ± {np.std(group_aucs):.4f}")
        print(f"      Best: {max(group_aucs):.4f}")

# All results
print("\n📊 All AUC Values:")
for result in roc_results:
    print(f"   {result['method']:35s}: {result['auc']:.4f}")

plt.close(fig)
print("\n✅ ROC curve generation complete!")