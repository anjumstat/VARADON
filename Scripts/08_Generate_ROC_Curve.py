# -*- coding: utf-8 -*-
"""
Generate REAL ROC Curves from actual predictions and true labels
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

base_dirs = [
    r"D:\zebfish\VARDON\VARDON_Results_Corrected_0.01",
    r"D:\zebfish\VARDON\VARDON_Results_Corrected_0.001",
    r"D:\zebfish\VARDON\VARDON_Results_Corrected_0.0001",
]

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

# Best configurations from your results
best_configs = {
    'VARDON_Light': {'lr': 0.0001, 'bs': 32},
    'VARDON_RealVD_Adaptive': {'lr': 0.0001, 'bs': 128},
    'Gaussian_Dropout_NN': {'lr': 0.0001, 'bs': 32},
    'VARDON_RealVD': {'lr': 0.0001, 'bs': 64},
    'Standard_Dropout': {'lr': 0.0001, 'bs': 32},
    'VARDON_AdaptiveVD': {'lr': 0.0001, 'bs': 128},
    'Logistic_Regression': {'lr': 0.001, 'bs': 64},
    'VARDON_Full': {'lr': 0.01, 'bs': 128}
}

method_colors = {
    'Logistic_Regression': '#808080',
    'Gaussian_Dropout_NN': '#1f77b4',
    'Standard_Dropout': '#ff7f0e',
    'VARDON_RealVD': '#2ca02c',
    'VARDON_AdaptiveVD': '#d62728',
    'VARDON_RealVD_Adaptive': '#9467bd',
    'VARDON_Full': '#8c564b',
    'VARDON_Light': '#e377c2'
}

def load_roc_data(method, lr, bs):
    """Load predictions and true labels from all folds to compute ROC"""
    
    all_probs = []
    all_labels = []
    
    for base_dir in base_dirs:
        if str(lr) in base_dir:
            # Find the config folder
            config_pattern = f"lr_*_bs_{bs}"
            config_dirs = glob.glob(os.path.join(base_dir, "cv_runs", config_pattern, method))
            
            for config_dir in config_dirs:
                npy_dir = os.path.join(config_dir, "npy_files")
                if os.path.exists(npy_dir):
                    for fold in range(1, 11):
                        # Load predictions (probability for class 1 - Enzyme)
                        pred_file = os.path.join(npy_dir, f"fold{fold}_predictions.npy")
                        labels_file = os.path.join(npy_dir, f"fold{fold}_true_labels.npy")
                        
                        if os.path.exists(pred_file) and os.path.exists(labels_file):
                            pred = np.load(pred_file)
                            labels = np.load(labels_file)
                            
                            # Get probability for class 1 (Enzyme)
                            if pred.shape[1] == 2:
                                probs = pred[:, 1]
                            else:
                                probs = pred
                            
                            all_probs.extend(probs)
                            all_labels.extend(labels)
    
    if len(all_probs) > 0:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc
    return None, None, None

# =============================================
# GENERATE REAL ROC CURVES
# =============================================

print("\n" + "="*60)
print("Generating REAL ROC Curves from Actual Data")
print("="*60)

fig, ax = plt.subplots(figsize=(10, 8))

for method in methods:
    config = best_configs[method]
    fpr, tpr, roc_auc = load_roc_data(method, config['lr'], config['bs'])
    
    if fpr is not None:
        ax.plot(fpr, tpr, linewidth=2, 
                color=method_colors.get(method, '#1f77b4'),
                label=f'{method} (AUC = {roc_auc:.4f})')
        print(f"  ✅ {method}: AUC = {roc_auc:.4f}")
    else:
        print(f"  ⚠️ {method}: No ROC data found")

# Diagonal line
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Random (AUC = 0.5)')

ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('Figure 2. Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

# Save
output_dir = r"D:\zebfish\VARDON\Paper_Figures"
png_dir = os.path.join(output_dir, "PNG")
tiff_dir = os.path.join(output_dir, "TIFF")
os.makedirs(png_dir, exist_ok=True)
os.makedirs(tiff_dir, exist_ok=True)

png_path = os.path.join(png_dir, "Figure2_ROC_Curves.png")
tiff_path = os.path.join(tiff_dir, "Figure2_ROC_Curves.tiff")
fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(tiff_path, dpi=300, bbox_inches='tight', facecolor='white', 
            format='tiff', pil_kwargs={"compression": "tiff_lzw"})
print(f"\n✅ Saved: {png_path}")
print(f"✅ Saved: {tiff_path}")
plt.close(fig)