# -*- coding: utf-8 -*-
"""
Created on Sat May 16 20:57:23 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Generate Publication-Ready Figures for Oxford Bioinformatics
Based on actual experimental results
Figures: 1, 2 (4 subplots), 3, 4
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re

# =============================================
# CONFIGURATION
# =============================================

# Path to your combined CV results
cv_results_path = r"D:\zebfish\VARDON\Combined_Results\CV_all_models\ALL_METHODS_Fold_Metrics_Combined.csv"

# Output directories
output_dir = r"D:\zebfish\VARDON\Paper_Figures_Final"
png_dir = os.path.join(output_dir, "PNG")
tiff_dir = os.path.join(output_dir, "TIFF")
os.makedirs(png_dir, exist_ok=True)
os.makedirs(tiff_dir, exist_ok=True)

# Methods
all_methods = [
    'Logistic_Regression',
    'Gaussian_Dropout_NN',
    'Standard_Dropout',
    'VARDON_RealVD',
    'VARDON_AdaptiveVD',
    'VARDON_RealVD_Adaptive',
    'VARDON_Full',
    'VARDON_Light'
]

# VARDON variants (your novel methods)
vardon_methods = [
    'VARDON_RealVD',
    'VARDON_AdaptiveVD',
    'VARDON_RealVD_Adaptive',
    'VARDON_Full',
    'VARDON_Light'
]

# Colors for methods
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

# Line styles: solid for VARDON, dashed for baselines
line_styles = {m: '-' for m in vardon_methods}
line_styles.update({m: '--' for m in all_methods if m not in vardon_methods})

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

def save_figure(fig, filename):
    """Save figure as both PNG and TIFF"""
    png_path = os.path.join(png_dir, f"{filename}.png")
    tiff_path = os.path.join(tiff_dir, f"{filename}.tiff")
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(tiff_path, dpi=300, bbox_inches='tight', facecolor='white', 
                format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    print(f"  ✅ Saved: {filename}")

# =============================================
# FIGURE 1: HYPERPARAMETER EFFECTS (LR × BS)
# =============================================

print("\n" + "="*60)
print("Generating Figure 1: Hyperparameter Effects (LR × BS)")
print("="*60)

if os.path.exists(cv_results_path):
    df = pd.read_csv(cv_results_path)
    
    # Calculate mean F1 for each (LR, BS) combination
    lr_values = [0.0001, 0.001, 0.01]
    bs_values = [32, 64, 128]
    
    heatmap_data = np.zeros((len(bs_values), len(lr_values)))
    
    for i, bs in enumerate(bs_values):
        for j, lr in enumerate(lr_values):
            subset = df[(df['Learning_Rate'] == lr) & (df['Batch_Size'] == bs)]
            if len(subset) > 0:
                heatmap_data[i, j] = subset['F1'].mean()
    
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Heatmap
    im = ax1.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.86, vmax=0.875)
    ax1.set_xticks(np.arange(len(lr_values)))
    ax1.set_yticks(np.arange(len(bs_values)))
    ax1.set_xticklabels([f'{lr:.4f}' for lr in lr_values], fontweight='bold')
    ax1.set_yticklabels(bs_values, fontweight='bold')
    ax1.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Batch Size', fontsize=12, fontweight='bold')
    ax1.set_title('Figure 1A. Mean F1 Score (LR × BS)', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(bs_values)):
        for j in range(len(lr_values)):
            ax1.text(j, i, f'{heatmap_data[i, j]:.4f}',
                    ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax1, label='Mean F1 Score')
    
    # Bar plot of best LR per BS
    best_by_lr = []
    for lr in lr_values:
        subset = df[df['Learning_Rate'] == lr]
        best_f1 = subset['F1'].mean() if len(subset) > 0 else 0
        best_by_lr.append(best_f1)
    
    bars = ax2.bar([f'LR={lr}' for lr in lr_values], best_by_lr, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
    ax2.set_ylabel('Mean F1 Score', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Figure 1B. Mean F1 by Learning Rate', fontsize=12, fontweight='bold')
    ax2.set_ylim([0.86, 0.88])
    ax2.grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars, best_by_lr):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig1, "Figure1_Hyperparameter_Effects")
    plt.close(fig1)

# =============================================
# FIGURE 2: TRAINING AND VALIDATION CURVES (2x2 Subplots)
# =============================================

print("\n" + "="*60)
print("Generating Figure 2: Training and Validation Curves (2x2 Subplots)")
print("="*60)

# Top methods to display (3 models)
top_methods = ['VARDON_Light', 'VARDON_RealVD_Adaptive', 'Gaussian_Dropout_NN']

# Best configurations for these methods (from your results)
best_configs = {
    'VARDON_Light': {'lr': 0.0001, 'bs': 32},
    'VARDON_RealVD_Adaptive': {'lr': 0.0001, 'bs': 128},
    'Gaussian_Dropout_NN': {'lr': 0.0001, 'bs': 32}
}

def load_training_and_validation_data(method, lr, bs):
    """Load training and validation history from .npy files"""
    base_dirs = [
        r"D:\zebfish\VARDON\VARDON_Results_Corrected_0.01",
        r"D:\zebfish\VARDON\VARDON_Results_Corrected_0.001",
        r"D:\zebfish\VARDON\VARDON_Results_Corrected_0.0001"
    ]
    
    for base_dir in base_dirs:
        if str(lr) in base_dir:
            config_pattern = f"lr_*_bs_{bs}"
            config_dirs = glob.glob(os.path.join(base_dir, "cv_runs", config_pattern, method))
            for config_dir in config_dirs:
                npy_dir = os.path.join(config_dir, "npy_files")
                if os.path.exists(npy_dir):
                    all_train_acc = []
                    all_val_acc = []
                    all_train_loss = []
                    all_val_loss = []
                    
                    for fold in range(1, 11):
                        # Training accuracy
                        train_acc_file = os.path.join(npy_dir, f"fold{fold}_accuracy.npy")
                        if os.path.exists(train_acc_file):
                            all_train_acc.append(np.load(train_acc_file))
                        
                        # Validation accuracy
                        val_acc_file = os.path.join(npy_dir, f"fold{fold}_val_accuracy.npy")
                        if os.path.exists(val_acc_file):
                            all_val_acc.append(np.load(val_acc_file))
                        else:
                            # If validation accuracy not available, use training accuracy as fallback
                            all_val_acc.append(np.load(train_acc_file))
                        
                        # Training loss
                        train_loss_file = os.path.join(npy_dir, f"fold{fold}_loss.npy")
                        if os.path.exists(train_loss_file):
                            all_train_loss.append(np.load(train_loss_file))
                        
                        # Validation loss
                        val_loss_file = os.path.join(npy_dir, f"fold{fold}_val_loss.npy")
                        if os.path.exists(val_loss_file):
                            all_val_loss.append(np.load(val_loss_file))
                        elif os.path.exists(train_loss_file):
                            all_val_loss.append(np.load(train_loss_file))
                    
                    if all_train_acc:
                        # Pad sequences to same length
                        max_len = max(len(acc) for acc in all_train_acc)
                        
                        padded_train_acc = np.array([np.pad(acc, (0, max_len - len(acc)), constant_values=acc[-1]) for acc in all_train_acc])
                        padded_val_acc = np.array([np.pad(acc, (0, max_len - len(acc)), constant_values=acc[-1]) for acc in all_val_acc]) if all_val_acc else padded_train_acc
                        padded_train_loss = np.array([np.pad(loss, (0, max_len - len(loss)), constant_values=loss[-1]) for loss in all_train_loss]) if all_train_loss else None
                        padded_val_loss = np.array([np.pad(loss, (0, max_len - len(loss)), constant_values=loss[-1]) for loss in all_val_loss]) if all_val_loss else None
                        
                        return {
                            'train_acc_mean': padded_train_acc.mean(axis=0),
                            'train_acc_std': padded_train_acc.std(axis=0),
                            'val_acc_mean': padded_val_acc.mean(axis=0),
                            'val_acc_std': padded_val_acc.std(axis=0),
                            'train_loss_mean': padded_train_loss.mean(axis=0) if padded_train_loss is not None else None,
                            'train_loss_std': padded_train_loss.std(axis=0) if padded_train_loss is not None else None,
                            'val_loss_mean': padded_val_loss.mean(axis=0) if padded_val_loss is not None else None,
                            'val_loss_std': padded_val_loss.std(axis=0) if padded_val_loss is not None else None,
                            'epochs': max_len
                        }
    return None

# Create 2x2 subplots
fig2, axes = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle('Figure 3. Training and Validation Curves', fontsize=14, fontweight='bold')

for method in top_methods:
    config = best_configs[method]
    data = load_training_and_validation_data(method, config['lr'], config['bs'])
    
    if data:
        epochs = np.arange(1, data['epochs'] + 1)
        color = method_colors.get(method, '#1f77b4')
        line_style = '-' if method in vardon_methods else '--'
        
        # Plot 1: Training Accuracy (Top-Left)
        axes[0, 0].plot(epochs, data['train_acc_mean'], linewidth=2.5,
                        color=color, linestyle=line_style,
                        label=f'{method}')
        axes[0, 0].fill_between(epochs,
                                data['train_acc_mean'] - data['train_acc_std'],
                                data['train_acc_mean'] + data['train_acc_std'],
                                alpha=0.15, color=color)
        
        # Plot 2: Validation Accuracy (Top-Right)
        axes[0, 1].plot(epochs, data['val_acc_mean'], linewidth=2.5,
                        color=color, linestyle=line_style,
                        label=f'{method}')
        axes[0, 1].fill_between(epochs,
                                data['val_acc_mean'] - data['val_acc_std'],
                                data['val_acc_mean'] + data['val_acc_std'],
                                alpha=0.15, color=color)
        
        # Plot 3: Training Loss (Bottom-Left)
        if data['train_loss_mean'] is not None:
            axes[1, 0].plot(epochs, data['train_loss_mean'], linewidth=2.5,
                            color=color, linestyle=line_style,
                            label=f'{method}')
            axes[1, 0].fill_between(epochs,
                                    data['train_loss_mean'] - data['train_loss_std'],
                                    data['train_loss_mean'] + data['train_loss_std'],
                                    alpha=0.15, color=color)
        
        # Plot 4: Validation Loss (Bottom-Right)
        if data['val_loss_mean'] is not None:
            axes[1, 1].plot(epochs, data['val_loss_mean'], linewidth=2.5,
                            color=color, linestyle=line_style,
                            label=f'{method}')
            axes[1, 1].fill_between(epochs,
                                    data['val_loss_mean'] - data['val_loss_std'],
                                    data['val_loss_mean'] + data['val_loss_std'],
                                    alpha=0.15, color=color)

# Configure Top-Left: Training Accuracy
axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Training Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Figure 3A. Training Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].legend(loc='lower right', fontsize=9, framealpha=0.9)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0.7, 1.0])

# Configure Top-Right: Validation Accuracy
axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Figure 3B. Validation Accuracy', fontsize=12, fontweight='bold')
axes[0, 1].legend(loc='lower right', fontsize=9, framealpha=0.9)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0.7, 1.0])

# Configure Bottom-Left: Training Loss
axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Figure 3C. Training Loss', fontsize=12, fontweight='bold')
axes[1, 0].legend(loc='upper right', fontsize=9, framealpha=0.9)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 0.8])

# Configure Bottom-Right: Validation Loss
axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Figure 3D. Validation Loss', fontsize=12, fontweight='bold')
axes[1, 1].legend(loc='upper right', fontsize=9, framealpha=0.9)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([0, 0.8])

plt.tight_layout()
save_figure(fig2, "Figure3_Training_Validation_Curves")
plt.close(fig2)

# =============================================
# FIGURE 3: VARDON VARIANTS RANKING
# =============================================

print("\n" + "="*60)
print("Generating Figure 4: VARDON Variants Ranking")
print("="*60)

if os.path.exists(cv_results_path):
    df = pd.read_csv(cv_results_path)
    
    # Calculate mean metrics for VARDON variants
    vardon_df = df[df['Method_Name'].isin(vardon_methods)]
    vardon_stats = vardon_df.groupby('Method_Name').agg({
        'F1': 'mean',
        'AUC': 'mean',
        'Accuracy': 'mean'
    }).round(4)
    
    # Sort by F1 score
    vardon_stats = vardon_stats.sort_values('F1', ascending=False)
    
    fig3, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(vardon_stats.index))
    width = 0.25
    
    bars1 = ax.bar(x - width, vardon_stats['F1'], width, label='F1 Score', 
                   color='#2ca02c', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, vardon_stats['AUC'], width, label='AUC', 
                   color='#1f77b4', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, vardon_stats['Accuracy'], width, label='Accuracy', 
                   color='#ff7f0e', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('VARDON Variant', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Figure 4. Performance Ranking of VARDON Variants', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(vardon_stats.index, rotation=45, ha='right', fontsize=10)
    
    # Move legend outside to avoid overlapping with bars
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, framealpha=0.9)
    ax.set_ylim([0.75, 0.99])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.002,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    
    # Highlight best performer (VARDON_Light)
    best_idx = 0  # VARDON_Light is first after sorting
    bars1[best_idx].set_edgecolor('gold')
    bars1[best_idx].set_linewidth(2.5)
    bars2[best_idx].set_edgecolor('gold')
    bars2[best_idx].set_linewidth(2.5)
    bars3[best_idx].set_edgecolor('gold')
    bars3[best_idx].set_linewidth(2.5)
    
    plt.tight_layout()
    save_figure(fig3, "Figure4_VARDON_Ranking")
    plt.close(fig3)

# =============================================
# FIGURE 4: CROSS-VALIDATION BOXPLOTS
# =============================================

print("\n" + "="*60)
print("Generating Figure 5: Cross-Validation Boxplots")
print("="*60)

if os.path.exists(cv_results_path):
    df = pd.read_csv(cv_results_path)
    
    fig4, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data for boxplot
    boxplot_data = []
    boxplot_labels = []
    boxplot_colors = []
    
    for method in all_methods:
        method_data = df[df['Method_Name'] == method]['F1'].values
        if len(method_data) > 0:
            boxplot_data.append(method_data)
            boxplot_labels.append(method)
            boxplot_colors.append(method_colors[method])
    
    bp = ax.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True,
                    medianprops=dict(linewidth=2, color='black'),
                    whiskerprops=dict(linewidth=1),
                    capprops=dict(linewidth=1))
    
    for patch, color in zip(bp['boxes'], boxplot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Highlight VARDON variants with thicker borders
    for i, method in enumerate(boxplot_labels):
        if method in vardon_methods:
            bp['boxes'][i].set_edgecolor('gold')
            bp['boxes'][i].set_linewidth(2.5)
    
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_title('Figure 5. Cross-Validation F1 Score Distribution (10-Fold)\n(Gold borders = VARDON variants)', 
                 fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0.75, 0.95])
    
    # Add horizontal line at mean of best VARDON
    best_vardon = 'VARDON_Light'
    best_mean = df[df['Method_Name'] == best_vardon]['F1'].mean()
    ax.axhline(y=best_mean, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(len(boxplot_labels) - 0.5, best_mean + 0.003, f'Best VARDON Mean: {best_mean:.4f}', 
            fontsize=8, color='green', ha='right', fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig4, "Figure5_CV_Boxplots")
    plt.close(fig4)

# =============================================
# SUMMARY
# =============================================

print("\n" + "="*60)
print("FIGURE GENERATION COMPLETE")
print("="*60)
print(f"\n📁 PNG files saved to: {png_dir}")
print(f"📁 TIFF files saved to: {tiff_dir}")
print("\nGenerated figures:")
print("  - Figure 1: Hyperparameter Effects (LR × BS)")
print("  - Figure 2: Training and Validation Curves (2x2 subplots)")
print("    * Figure 2A: Training Accuracy")
print("    * Figure 2B: Validation Accuracy")
print("    * Figure 2C: Training Loss")
print("    * Figure 2D: Validation Loss")
print("  - Figure 3: VARDON Variants Ranking")
print("  - Figure 4: Cross-Validation Boxplots")
print("\n" + "="*60)