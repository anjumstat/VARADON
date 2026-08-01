# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 23:58:45 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Generate Publication-Ready Figures for Oxford Bioinformatics
Based on REVISION results (13 methods, all in one directory)
Figures: 1, 2 (4 subplots), 3, 4
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import ast

# =============================================
# CONFIGURATION
# =============================================

# UPDATED: Path to your combined results (revision)
cv_results_path = r"D:\zebfish\revision\Combined_Results\Combined_CV_Results.csv"
test_results_path = r"D:\zebfish\revision\Combined_Results\Combined_Test_Results_All_Models.csv"

# UPDATED: Output directories
output_dir = r"D:\zebfish\revision\Paper_Figures"
png_dir = os.path.join(output_dir, "PNG")
tiff_dir = os.path.join(output_dir, "TIFF")
svg_dir = os.path.join(output_dir, "SVG")
for d in [png_dir, tiff_dir, svg_dir]:
    os.makedirs(d, exist_ok=True)

# UPDATED: All 13 methods
all_methods = [
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

# VARDON variants (novel methods)
vardon_methods = [
    'VARDON_RealVD',
    'VARDON_AdaptiveVD',
    'VARDON_RealVD_Adaptive',
    'VARDON_Full',
    'VARDON_Light'
]

# UPDATED: All VARDON variants + ablations
all_vardon_methods = [
    'VARDON_RealVD',
    'VARDON_AdaptiveVD',
    'VARDON_RealVD_Adaptive',
    'VARDON_Full',
    'VARDON_Light',
    'VARDON_Full_No_Gate',
    'VARDON_Full_No_Sparsity',
    'VARDON_Full_No_Both'
]

# UPDATED: Colors for all 13 methods
method_colors = {
    'Logistic_Regression': '#808080',
    'Gaussian_Dropout_NN': '#1f77b4',
    'Standard_Dropout': '#ff7f0e',
    'MLP_BatchNorm': '#2ca02c',
    'Residual_MLP': '#17becf',
    'VARDON_RealVD': '#9467bd',
    'VARDON_AdaptiveVD': '#d62728',
    'VARDON_RealVD_Adaptive': '#e377c2',
    'VARDON_Full': '#8c564b',
    'VARDON_Light': '#bcbd22',
    'VARDON_Full_No_Gate': '#7f7f7f',
    'VARDON_Full_No_Sparsity': '#f7b6d2',
    'VARDON_Full_No_Both': '#98df8a'
}

# Line styles: solid for VARDON, dashed for baselines, dotted for ablations
line_styles = {m: '-' for m in all_vardon_methods}
line_styles.update({m: '--' for m in all_methods if m not in all_vardon_methods})

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
    """Save figure as PNG, TIFF, and SVG"""
    png_path = os.path.join(png_dir, f"{filename}.png")
    tiff_path = os.path.join(tiff_dir, f"{filename}.tiff")
    svg_path = os.path.join(svg_dir, f"{filename}.svg")
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(tiff_path, dpi=300, bbox_inches='tight', facecolor='white', 
                format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
    print(f"  ✅ Saved: {filename}")

# =============================================
# FIGURE 1: HYPERPARAMETER EFFECTS (LR × BS)
# =============================================

print("\n" + "="*60)
print("Generating Figure 1: Hyperparameter Effects (LR × BS)")
print("="*60)

# Load data
if os.path.exists(cv_results_path):
    df = pd.read_csv(cv_results_path)
    
    # Check column names
    print(f"Columns: {df.columns.tolist()}")
    
    # Use correct column names
    lr_col = 'learning_rate'
    bs_col = 'batch_size'
    method_col = 'method'
    
    # Calculate mean F1 for each (LR, BS) combination across all methods
    lr_values = [0.0001, 0.001, 0.01]
    bs_values = [32, 64, 128]
    
    heatmap_data = np.zeros((len(bs_values), len(lr_values)))
    
    for i, bs in enumerate(bs_values):
        for j, lr in enumerate(lr_values):
            subset = df[(df[lr_col] == lr) & (df[bs_col] == bs)]
            if len(subset) > 0 and 'mean_f1' in df.columns:
                heatmap_data[i, j] = subset['mean_f1'].mean()
    
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Heatmap
    im = ax1.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.83, vmax=0.87)
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
        subset = df[df[lr_col] == lr]
        if 'mean_f1' in df.columns:
            best_f1 = subset['mean_f1'].mean() if len(subset) > 0 else 0
        else:
            best_f1 = 0
        best_by_lr.append(best_f1)
    
    bars = ax2.bar([f'LR={lr}' for lr in lr_values], best_by_lr, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
    ax2.set_ylabel('Mean F1 Score', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Figure 1B. Mean F1 by Learning Rate', fontsize=12, fontweight='bold')
    ax2.set_ylim([0.83, 0.88])
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
print("Generating Figure 3: Training and Validation Curves (2x2 Subplots)")
print("="*60)

# Top methods to display (3 models)
top_methods = ['VARDON_Light', 'VARDON_RealVD_Adaptive', 'Gaussian_Dropout_NN']

# Best configurations for these methods (from revision results)
best_configs = {
    'VARDON_Light': {'lr': 0.01, 'bs': 64},
    'VARDON_RealVD_Adaptive': {'lr': 0.01, 'bs': 32},
    'Gaussian_Dropout_NN': {'lr': 0.01, 'bs': 32}
}

def load_training_and_validation_data(method, lr, bs):
    """Load training and validation history from .npy files"""
    base_dir = r"D:\zebfish\revision\VARDON_Results_Corrected_0.01"
    
    # Convert LR to folder format
    lr_str = f"{lr:.4f}"
    lr_parts = lr_str.split('.')
    if len(lr_parts) == 2:
        frac = lr_parts[1].ljust(5, '0')[:5]
        lr_folder = f"lr_{lr_parts[0]}_{frac}"
    else:
        lr_folder = f"lr_0_{int(lr*100000):05d}"
    
    config_dir = os.path.join(base_dir, "cv_runs", f"{lr_folder}_bs_{bs}", method)
    npy_dir = os.path.join(config_dir, "npy_files")
    
    if not os.path.exists(npy_dir):
        print(f"    ⚠️ Directory not found: {npy_dir}")
        return None
    
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
        elif os.path.exists(train_acc_file):
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
    print(f"\n📊 Loading data for: {method} (LR={config['lr']}, BS={config['bs']})")
    data = load_training_and_validation_data(method, config['lr'], config['bs'])
    
    if data:
        epochs = np.arange(1, data['epochs'] + 1)
        color = method_colors.get(method, '#1f77b4')
        line_style = '-' if method in all_vardon_methods else '--'
        
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
        print(f"    ✅ Data loaded ({data['epochs']} epochs)")

# Configure axes
axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Training Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Figure 3A. Training Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].legend(loc='lower right', fontsize=9, framealpha=0.9)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0.7, 1.0])

axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Figure 3B. Validation Accuracy', fontsize=12, fontweight='bold')
axes[0, 1].legend(loc='lower right', fontsize=9, framealpha=0.9)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0.7, 1.0])

axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Figure 3C. Training Loss', fontsize=12, fontweight='bold')
axes[1, 0].legend(loc='upper right', fontsize=9, framealpha=0.9)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 0.8])

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
# FIGURE 3: VARDON VARIANTS RANKING (UPDATED)
# =============================================
# =============================================
# FIGURE 2: TRAINING AND VALIDATION CURVES (2x2 Subplots) - 5 MODELS
# =============================================

print("\n" + "="*60)
print("Generating Figure 3: Training and Validation Curves (2x2 Subplots) - 5 Models")
print("="*60)

# UPDATED: Top 5 methods to display
top_methods = [
    'VARDON_Light',           # Best lightweight VARDON
    'VARDON_RealVD_Adaptive', # Best statistical performer (11 wins)
    'VARDON_Full_No_Sparsity',# Best LOSO + best AUC
    'Residual_MLP',           # Best test MCC (baseline)
    'Gaussian_Dropout_NN'     # Best CV-selected baseline
]

# Best configurations for these methods (from revision results)
best_configs = {
    'VARDON_Light': {'lr': 0.01, 'bs': 64},
    'VARDON_RealVD_Adaptive': {'lr': 0.01, 'bs': 32},
    'VARDON_Full_No_Sparsity': {'lr': 0.01, 'bs': 32},
    'Residual_MLP': {'lr': 0.01, 'bs': 32},
    'Gaussian_Dropout_NN': {'lr': 0.01, 'bs': 32}
}

# UPDATED: Line styles and colors
method_line_styles = {
    'VARDON_Light': ('-', '#bcbd22'),           # Olive - solid
    'VARDON_RealVD_Adaptive': ('-', '#e377c2'), # Pink - solid
    'VARDON_Full_No_Sparsity': ('-', '#f7b6d2'),# Light Pink - solid
    'Residual_MLP': ('--', '#17becf'),          # Cyan - dashed (baseline)
    'Gaussian_Dropout_NN': ('--', '#1f77b4')    # Blue - dashed (baseline)
}

def load_training_and_validation_data(method, lr, bs):
    """Load training and validation history from .npy files"""
    base_dir = r"D:\zebfish\revision\VARDON_Results_Corrected_0.01"
    
    # Convert LR to folder format
    lr_str = f"{lr:.4f}"
    lr_parts = lr_str.split('.')
    if len(lr_parts) == 2:
        frac = lr_parts[1].ljust(5, '0')[:5]
        lr_folder = f"lr_{lr_parts[0]}_{frac}"
    else:
        lr_folder = f"lr_0_{int(lr*100000):05d}"
    
    config_dir = os.path.join(base_dir, "cv_runs", f"{lr_folder}_bs_{bs}", method)
    npy_dir = os.path.join(config_dir, "npy_files")
    
    if not os.path.exists(npy_dir):
        print(f"    ⚠️ Directory not found: {npy_dir}")
        return None
    
    all_train_acc = []
    all_val_acc = []
    all_train_loss = []
    all_val_loss = []
    
    for fold in range(1, 11):
        train_acc_file = os.path.join(npy_dir, f"fold{fold}_accuracy.npy")
        if os.path.exists(train_acc_file):
            all_train_acc.append(np.load(train_acc_file))
        
        val_acc_file = os.path.join(npy_dir, f"fold{fold}_val_accuracy.npy")
        if os.path.exists(val_acc_file):
            all_val_acc.append(np.load(val_acc_file))
        elif os.path.exists(train_acc_file):
            all_val_acc.append(np.load(train_acc_file))
        
        train_loss_file = os.path.join(npy_dir, f"fold{fold}_loss.npy")
        if os.path.exists(train_loss_file):
            all_train_loss.append(np.load(train_loss_file))
        
        val_loss_file = os.path.join(npy_dir, f"fold{fold}_val_loss.npy")
        if os.path.exists(val_loss_file):
            all_val_loss.append(np.load(val_loss_file))
        elif os.path.exists(train_loss_file):
            all_val_loss.append(np.load(train_loss_file))
    
    if all_train_acc:
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
fig2.suptitle('Figure 3. Training and Validation Curves (5 Models)', fontsize=14, fontweight='bold')

# Dictionary to store legend labels
legend_labels = []

for method in top_methods:
    config = best_configs[method]
    print(f"\n📊 Loading data for: {method} (LR={config['lr']}, BS={config['bs']})")
    data = load_training_and_validation_data(method, config['lr'], config['bs'])
    
    if data:
        epochs = np.arange(1, data['epochs'] + 1)
        linestyle, color = method_line_styles[method]
        
        # Determine if VARDON (solid) or baseline (dashed)
        is_vardon = method in ['VARDON_Light', 'VARDON_RealVD_Adaptive', 'VARDON_Full_No_Sparsity']
        label = method + (' (VARDON)' if is_vardon else ' (Baseline)')
        
        # Plot 1: Training Accuracy (Top-Left)
        axes[0, 0].plot(epochs, data['train_acc_mean'], linewidth=2.5,
                        color=color, linestyle=linestyle,
                        label=label)
        axes[0, 0].fill_between(epochs,
                                data['train_acc_mean'] - data['train_acc_std'],
                                data['train_acc_mean'] + data['train_acc_std'],
                                alpha=0.12, color=color)
        
        # Plot 2: Validation Accuracy (Top-Right)
        axes[0, 1].plot(epochs, data['val_acc_mean'], linewidth=2.5,
                        color=color, linestyle=linestyle,
                        label=label)
        axes[0, 1].fill_between(epochs,
                                data['val_acc_mean'] - data['val_acc_std'],
                                data['val_acc_mean'] + data['val_acc_std'],
                                alpha=0.12, color=color)
        
        # Plot 3: Training Loss (Bottom-Left)
        if data['train_loss_mean'] is not None:
            axes[1, 0].plot(epochs, data['train_loss_mean'], linewidth=2.5,
                            color=color, linestyle=linestyle,
                            label=label)
            axes[1, 0].fill_between(epochs,
                                    data['train_loss_mean'] - data['train_loss_std'],
                                    data['train_loss_mean'] + data['train_loss_std'],
                                    alpha=0.12, color=color)
        
        # Plot 4: Validation Loss (Bottom-Right)
        if data['val_loss_mean'] is not None:
            axes[1, 1].plot(epochs, data['val_loss_mean'], linewidth=2.5,
                            color=color, linestyle=linestyle,
                            label=label)
            axes[1, 1].fill_between(epochs,
                                    data['val_loss_mean'] - data['val_loss_std'],
                                    data['val_loss_mean'] + data['val_loss_std'],
                                    alpha=0.12, color=color)
        print(f"    ✅ Data loaded ({data['epochs']} epochs)")

# Configure axes
axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Training Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Figure 3A. Training Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].legend(loc='lower right', fontsize=8, framealpha=0.9)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0.7, 1.0])

axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Figure 3B. Validation Accuracy', fontsize=12, fontweight='bold')
axes[0, 1].legend(loc='lower right', fontsize=8, framealpha=0.9)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0.7, 1.0])

axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Figure 3C. Training Loss', fontsize=12, fontweight='bold')
axes[1, 0].legend(loc='upper right', fontsize=8, framealpha=0.9)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 0.8])

axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Figure 3D. Validation Loss', fontsize=12, fontweight='bold')
axes[1, 1].legend(loc='upper right', fontsize=8, framealpha=0.9)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([0, 0.8])

plt.tight_layout()
save_figure(fig2, "Figure3_Training_Validation_Curves_5Models")
plt.close(fig2)


# =============================================
# FIGURE 4: CROSS-VALIDATION BOXPLOTS (UPDATED)
# =============================================

print("\n" + "="*60)
print("Generating Figure 5: Cross-Validation Boxplots")
print("="*60)

if os.path.exists(cv_results_path):
    df = pd.read_csv(cv_results_path)
    
    # Use 'mean_mcc' for boxplots
    metric_col = 'mean_mcc'
    
    fig4, ax = plt.subplots(figsize=(14, 7))
    
    # Prepare data for boxplot
    boxplot_data = []
    boxplot_labels = []
    boxplot_colors = []
    
    for method in all_methods:
        method_data = df[df['method'] == method][metric_col].values
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
        if method in all_vardon_methods:
            bp['boxes'][i].set_edgecolor('gold')
            bp['boxes'][i].set_linewidth(2.5)
    
    ax.set_ylabel('Mean MCC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_title('Figure 5. Cross-Validation MCC Distribution (10-Fold)\n(Gold borders = VARDON variants)', 
                 fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0.70, 0.90])
    
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
print(f"📁 SVG files saved to: {svg_dir}")
print("\nGenerated figures:")
print("  - Figure 1: Hyperparameter Effects (LR × BS)")
print("  - Figure 3: Training and Validation Curves (2x2 subplots)")
print("    * Figure 3A: Training Accuracy")
print("    * Figure 3B: Validation Accuracy")
print("    * Figure 3C: Training Loss")
print("    * Figure 3D: Validation Loss")
print("  - Figure 4: VARDON Variants Ranking")
print("  - Figure 5: Cross-Validation Boxplots")
print("  - Figure 2: ROC Curves (separate script)")
print("\n" + "="*60)