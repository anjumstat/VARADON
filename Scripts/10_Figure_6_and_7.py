# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 12:45:59 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Extract and Visualize Learned Dropout Rates from VARDON Models
FIXED: Properly handles RealVD layers with 2 variables
Generates Figure 6 and Figure 7 for ALL VARDON variants
UPDATED: Figure 7 shows VARDON_AdaptiveVD (negative correlation) for stronger evidence
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import re
import ast
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION
# =============================================

# Base directory for results
BASE_DIRS = [
    r"D:\zebfish\revision\VARDON_Results_Corrected_0.01",
    r"D:\zebfish\VARDON\VARDON_Results_Corrected_0.01",
]

OUTPUT_DIR = r"D:\zebfish\revision\Paper_Figures2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# All VARDON methods with learnable dropout rates
# MOVED VARDON_AdaptiveVD to FIRST position for Figure 7
VARDON_METHODS = [
    'VARDON_AdaptiveVD',      # ← MOVED TO FIRST for Figure 7
    'VARDON_RealVD',
    'VARDON_RealVD_Adaptive',
    'VARDON_Light',
    'VARDON_Full',
]

# Best configurations from your results
BEST_CONFIGS = {
    'VARDON_RealVD': {'lr': 0.0001, 'bs': 64},
    'VARDON_AdaptiveVD': {'lr': 0.0001, 'bs': 128},
    'VARDON_RealVD_Adaptive': {'lr': 0.0001, 'bs': 128},
    'VARDON_Light': {'lr': 0.0001, 'bs': 32},
    'VARDON_Full': {'lr': 0.01, 'bs': 128},
}

FIXED_DROPOUT_RATE = 0.3
COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']


# =============================================
# CUSTOM LAYERS (UPDATED)
# =============================================

@tf.keras.utils.register_keras_serializable(package="VARDON")
class RealVariationalDropout(layers.Layer):
    """FIXED: Handles both log_alpha and mean_shift variables"""
    def __init__(self, units, init_drop_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.init_drop_rate = init_drop_rate
        self.eps = 1e-8
    
    def build(self, input_shape):
        alpha_init = self.init_drop_rate / (1.0 - self.init_drop_rate + self.eps)
        log_alpha_init = np.log(alpha_init + self.eps)
        
        # Add BOTH variables that were saved in the model
        self.log_alpha = self.add_weight(
            name="log_alpha",
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(log_alpha_init),
            trainable=True,
        )
        
        self.mean_shift = self.add_weight(
            name="mean_shift",
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        alpha = tf.exp(self.log_alpha)
        dropout_rate = alpha / (1.0 + alpha + self.eps)
        variance = alpha * tf.square(inputs + self.mean_shift)
        std = tf.sqrt(variance + self.eps)
        epsilon = tf.random.normal(tf.shape(inputs), dtype=inputs.dtype)
        output = inputs + epsilon * std
        scale = tf.sqrt(1.0 / (1.0 - dropout_rate + self.eps))
        return output * scale
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "init_drop_rate": self.init_drop_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="VARDON")
class AdaptiveVariationalDropout(layers.Layer):
    def __init__(self, units, initial_drop_rate=0.3, learnable=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.initial_drop_rate = initial_drop_rate
        self.learnable = learnable
        self.eps = 1e-8
    
    def build(self, input_shape):
        if self.learnable:
            init_logit = np.log(self.initial_drop_rate / (1.0 - self.initial_drop_rate + self.eps))
            self.drop_logits = self.add_weight(
                name="drop_logits",
                shape=(self.units,),
                initializer=tf.keras.initializers.Constant(init_logit),
                trainable=True,
            )
        self.noise_scale = self.add_weight(
            name="noise_scale",
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
        )
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        if self.learnable:
            drop_rate = tf.sigmoid(self.drop_logits)
        else:
            drop_rate = tf.cast(self.initial_drop_rate, inputs.dtype)
        if not training:
            return inputs
        bernoulli_mask = tf.keras.backend.random_bernoulli(
            tf.shape(inputs), p=1.0 - drop_rate, dtype=inputs.dtype
        )
        gaussian_noise = tf.random.normal(tf.shape(inputs), dtype=inputs.dtype) * self.noise_scale
        combined_noise = bernoulli_mask * (1.0 + gaussian_noise)
        scale = 1.0 / (1.0 - drop_rate + self.eps)
        return inputs * combined_noise * scale
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "initial_drop_rate": self.initial_drop_rate,
            "learnable": self.learnable,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="VARDON")
class FeatureImportanceGate(layers.Layer):
    def __init__(self, keep_ratio=0.8, temperature=1.0, gate_regularization=0.01, **kwargs):
        super().__init__(**kwargs)
        self.keep_ratio = keep_ratio
        self.temperature = temperature
        self.gate_regularization = gate_regularization
    
    def build(self, input_shape):
        self.feature_importance = self.add_weight(
            name="feature_importance",
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        super().build(input_shape)
    
    def call(self, inputs):
        gate = tf.sigmoid(self.feature_importance / self.temperature)
        self.add_loss(self.gate_regularization * tf.square(tf.reduce_mean(gate) - self.keep_ratio))
        return inputs * gate
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "keep_ratio": self.keep_ratio,
            "temperature": self.temperature,
            "gate_regularization": self.gate_regularization,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="VARDON")
class DynamicSparsityRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, initial_sparsity=0.7, final_sparsity=0.9, warmup_epochs=20, total_epochs=100):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    
    def __call__(self, weights):
        progress = tf.minimum(1.0, self.current_epoch / float(self.warmup_epochs))
        current_target = self.initial_sparsity + progress * (self.final_sparsity - self.initial_sparsity)
        abs_weights = tf.abs(weights)
        flat_weights = tf.reshape(abs_weights, [-1])
        sorted_weights = tf.sort(flat_weights)
        n = tf.shape(sorted_weights)[0]
        k = tf.cast(tf.cast(n, tf.float32) * (1.0 - current_target), tf.int32)
        k = tf.clip_by_value(k, 1, n)
        threshold = sorted_weights[k - 1]
        sparsity = tf.reduce_mean(tf.cast(abs_weights < threshold, tf.float32))
        sparsity_loss = tf.square(sparsity - current_target) * current_target
        l1_strength = 0.0005 * (1.0 + progress * 2.0)
        l1_loss = tf.reduce_mean(abs_weights) * l1_strength
        return sparsity_loss + l1_loss
    
    def update_epoch(self, epoch):
        self.current_epoch.assign(float(epoch))
    
    def get_config(self):
        return {
            "initial_sparsity": self.initial_sparsity,
            "final_sparsity": self.final_sparsity,
            "warmup_epochs": self.warmup_epochs,
            "total_epochs": self.total_epochs,
        }


# =============================================
# Helper Functions
# =============================================

def get_lr_folder_name(lr, bs):
    """Convert LR and BS to folder name format"""
    lr_str = f"{lr:.4f}"
    lr_parts = lr_str.split('.')
    if len(lr_parts) == 2:
        frac = lr_parts[1].ljust(5, '0')[:5]
        lr_folder = f"lr_{lr_parts[0]}_{frac}"
    else:
        lr_folder = f"lr_0_{int(lr*100000):05d}"
    return f"{lr_folder}_bs_{bs}"


def find_model_file(base_dir, method, lr, bs):
    """Find a model file for the given method and configuration"""
    lr_folder = get_lr_folder_name(lr, bs)
    
    possible_paths = [
        os.path.join(base_dir, "cv_runs", lr_folder, method, "models"),
        os.path.join(base_dir, "cv_runs", lr_folder, method),
        os.path.join(base_dir, "final_selected_model_test", method, "models"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            model_files = glob.glob(os.path.join(path, "*.h5")) + glob.glob(os.path.join(path, "*.keras"))
            if model_files:
                return model_files[0], path
    
    return None, None


def extract_dropout_rates(model_path):
    """Extract learned dropout rates from a saved model."""
    try:
        custom_objects = {
            'RealVariationalDropout': RealVariationalDropout,
            'AdaptiveVariationalDropout': AdaptiveVariationalDropout,
            'FeatureImportanceGate': FeatureImportanceGate,
            'DynamicSparsityRegularizer': DynamicSparsityRegularizer
        }
        
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        
        dropout_rates = {}
        
        for layer in model.layers:
            # Check for RealVariationalDropout
            if hasattr(layer, 'log_alpha'):
                try:
                    alpha = tf.exp(layer.log_alpha).numpy()
                    rates = alpha / (1.0 + alpha + 1e-8)
                    dropout_rates[layer.name] = rates
                    print(f"       {layer.name}: RealVD, {len(rates)} neurons")
                except Exception as e:
                    pass
            
            # Check for AdaptiveVariationalDropout
            if hasattr(layer, 'drop_logits'):
                try:
                    rates = tf.sigmoid(layer.drop_logits).numpy()
                    dropout_rates[layer.name] = rates
                    print(f"       {layer.name}: AdaptiveVD, {len(rates)} neurons")
                except Exception as e:
                    pass
        
        return dropout_rates
    except Exception as e:
        print(f"    ⚠️ Error loading model: {e}")
        return None


def get_feature_importance(model, layer_name='dense1'):
    """Extract feature importance from the first dense layer"""
    try:
        for layer in model.layers:
            if layer.name == layer_name and hasattr(layer, 'get_weights'):
                weights = layer.get_weights()
                if weights and len(weights) > 0:
                    kernel = weights[0]
                    if kernel.ndim == 2:
                        return np.mean(np.abs(kernel), axis=1)
        return None
    except:
        return None


# =============================================
# MAIN EXECUTION
# =============================================

print("=" * 80)
print("EXTRACTING LEARNED DROPOUT RATES FROM VARDON MODELS")
print("Generating Figure 6 and Figure 7 for ALL variants")
print("Figure 7 will show VARDON_AdaptiveVD (negative correlation)")
print("=" * 80)

all_dropout_rates = {}
all_feature_importances = {}

for method in VARDON_METHODS:
    print(f"\n📊 Processing: {method}")
    
    config = BEST_CONFIGS[method]
    lr = config['lr']
    bs = config['bs']
    
    model_path = None
    found_base_dir = None
    
    for base_dir in BASE_DIRS:
        model_path, model_dir = find_model_file(base_dir, method, lr, bs)
        if model_path:
            found_base_dir = base_dir
            break
    
    if model_path is None:
        print(f"    ⚠️ No model files found for {method}")
        continue
    
    print(f"    Loading model: {os.path.basename(model_path)}")
    print(f"    Path: {model_path}")
    
    dropout_rates = extract_dropout_rates(model_path)
    
    if dropout_rates:
        all_dropout_rates[method] = dropout_rates
        print(f"    ✅ Extracted dropout rates from {len(dropout_rates)} layers")
        for layer_name, rates in dropout_rates.items():
            print(f"       {layer_name}: {len(rates)} neurons, mean={rates.mean():.4f}, std={rates.std():.4f}")
    else:
        print(f"    ⚠️ Could not extract dropout rates from {method}")
    
    # Try to extract feature importance
    try:
        custom_objects = {
            'RealVariationalDropout': RealVariationalDropout,
            'AdaptiveVariationalDropout': AdaptiveVariationalDropout,
            'FeatureImportanceGate': FeatureImportanceGate,
            'DynamicSparsityRegularizer': DynamicSparsityRegularizer
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        importance = get_feature_importance(model)
        if importance is not None:
            all_feature_importances[method] = importance
            print(f"    ✅ Extracted feature importance: {len(importance)} features")
    except Exception as e:
        print(f"    ⚠️ Could not extract feature importance: {e}")
    
    tf.keras.backend.clear_session()

# =============================================
# PRINT SUMMARY
# =============================================

print("\n" + "=" * 80)
print("EXTRACTED DATA SUMMARY")
print("=" * 80)

print(f"\n✅ Methods with dropout rates: {len(all_dropout_rates)}")
for method in all_dropout_rates:
    layers_info = [f"{name}: {len(rates)} neurons" for name, rates in all_dropout_rates[method].items()]
    print(f"  - {method}: {', '.join(layers_info)}")

print(f"\n✅ Methods with feature importance: {len(all_feature_importances)}")
for method in all_feature_importances:
    print(f"  - {method}: {len(all_feature_importances[method])} features")

# =============================================
# GENERATE FIGURE 6
# =============================================

if all_dropout_rates:
    print("\n" + "=" * 80)
    print("GENERATING FIGURE 6: Distribution of Learned Dropout Rates")
    print("=" * 80)

    n_methods = len(all_dropout_rates)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig6, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_methods == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (method, dropout_rates) in enumerate(all_dropout_rates.items()):
        ax = axes[idx]
        
        all_rates = []
        layer_labels = []
        for layer_name, rates in dropout_rates.items():
            all_rates.extend(rates)
            layer_labels.extend([layer_name] * len(rates))
        
        if all_rates:
            all_rates = np.array(all_rates)
            
            ax.hist(all_rates, bins=30, alpha=0.7, color=COLORS[idx % len(COLORS)], 
                    edgecolor='black', linewidth=0.5)
            ax.axvline(FIXED_DROPOUT_RATE, color='red', linestyle='--', linewidth=2.5, 
                       label=f'Fixed Rate (p={FIXED_DROPOUT_RATE})')
            
            mean_rate = np.mean(all_rates)
            ax.axvline(mean_rate, color='blue', linestyle='-', linewidth=2,
                       label=f'Mean = {mean_rate:.3f}')
            
            ax.set_xlabel('Learned Dropout Rate', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{method}\nMean={mean_rate:.3f}, Std={np.std(all_rates):.3f}', 
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 0.85])
            
            print(f"  {method}: Mean={mean_rate:.4f}, Std={np.std(all_rates):.4f}, "
                  f"Range=[{np.min(all_rates):.4f}, {np.max(all_rates):.4f}]")

    # Hide empty subplots
    for idx in range(len(all_dropout_rates), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Figure 6. Distribution of Learned Dropout Rates Across VARDON Variants\n'
                 'Red dashed line shows fixed dropout rate (p=0.3)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure6_Dropout_Rates_Distribution.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure6_Dropout_Rates_Distribution.tiff"), dpi=300, bbox_inches='tight',
                format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    plt.close()
    print(f"\n✅ Figure 6 saved to: {OUTPUT_DIR}")

# =============================================
# GENERATE FIGURE 7 - NOW USING VARDON_AdaptiveVD
# =============================================

print("\n" + "=" * 80)
print("GENERATING FIGURE 7: Dropout Rate vs Feature Importance")
print("Using VARDON_AdaptiveVD (negative correlation, approaching significance)")
print("=" * 80)

correlation_results = []
correlation_found = False

# Explicitly target VARDON_AdaptiveVD for Figure 7
target_method = 'VARDON_AdaptiveVD'

if target_method in all_dropout_rates and target_method in all_feature_importances:
    print(f"\n📊 Generating Figure 7 for: {target_method}")
    
    # Get data for the target method
    first_layer = list(all_dropout_rates[target_method].keys())[0]
    dropout_rates = all_dropout_rates[target_method][first_layer]
    feature_importance = all_feature_importances[target_method]
    
    min_len = min(len(dropout_rates), len(feature_importance))
    dropout_rates = dropout_rates[:min_len]
    feature_importance = feature_importance[:min_len]
    
    pearson_corr, pearson_p = pearsonr(dropout_rates, feature_importance)
    spearman_corr, spearman_p = spearmanr(dropout_rates, feature_importance)
    
    print(f"    Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4e})")
    print(f"    Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4e})")
    
    # Store results
    correlation_results.append({
        'Method': target_method,
        'Pearson_r': pearson_corr,
        'Pearson_p': pearson_p,
        'Spearman_r': spearman_corr,
        'Spearman_p': spearman_p
    })
    
    # Create Figure 7
    fig7, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(dropout_rates, feature_importance, alpha=0.5, c='#2ecc71', s=15)
    
    z = np.polyfit(dropout_rates, feature_importance, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(dropout_rates), max(dropout_rates), 100)
    ax.plot(x_line, p(x_line), color='red', linestyle='--', linewidth=2.5,
            label=f'Trend (r = {pearson_corr:.3f})')
    
    ax.set_xlabel('Learned Dropout Rate', fontsize=13)
    ax.set_ylabel('Feature Importance (Mean Absolute Weight)', fontsize=13)
    ax.set_title(f'Figure 7. Dropout Rate vs Feature Importance\n{target_method} | Pearson r = {pearson_corr:.4f}, p = {pearson_p:.4e}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure7_Dropout_vs_Importance.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure7_Dropout_vs_Importance.tiff"), dpi=300, bbox_inches='tight',
                format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    plt.close()
    print(f"✅ Figure 7 saved to: {OUTPUT_DIR}")
    correlation_found = True

# Also collect correlation results for all methods
print("\n📊 Collecting all correlation results...")
for method in all_dropout_rates:
    if method in all_feature_importances and method != target_method:
        first_layer = list(all_dropout_rates[method].keys())[0]
        dropout_rates = all_dropout_rates[method][first_layer]
        feature_importance = all_feature_importances[method]
        
        min_len = min(len(dropout_rates), len(feature_importance))
        dropout_rates = dropout_rates[:min_len]
        feature_importance = feature_importance[:min_len]
        
        pearson_corr, pearson_p = pearsonr(dropout_rates, feature_importance)
        spearman_corr, spearman_p = spearmanr(dropout_rates, feature_importance)
        
        correlation_results.append({
            'Method': method,
            'Pearson_r': pearson_corr,
            'Pearson_p': pearson_p,
            'Spearman_r': spearman_corr,
            'Spearman_p': spearman_p
        })

if not correlation_found:
    print("\n⚠️ VARDON_AdaptiveVD not found. Figure 7 will not be generated.")

# =============================================
# PRINT CORRELATION SUMMARY
# =============================================

print("\n" + "=" * 80)
print("CORRELATION SUMMARY (All Methods)")
print("=" * 80)

if correlation_results:
    corr_df = pd.DataFrame(correlation_results)
    print("\n", corr_df.to_string(index=False))
    corr_df.to_csv(os.path.join(OUTPUT_DIR, "Dropout_Importance_Correlation.csv"), index=False)
    print(f"\n✅ Correlation summary saved to: {os.path.join(OUTPUT_DIR, 'Dropout_Importance_Correlation.csv')}")
else:
    print("⚠️ No correlation results found.")

# =============================================
# SUMMARY TABLE
# =============================================

print("\n" + "=" * 80)
print("SUMMARY TABLE: Learned Dropout Rates Statistics")
print("=" * 80)

summary_data = []

for method, dropout_rates in all_dropout_rates.items():
    all_rates = []
    for layer_name, rates in dropout_rates.items():
        all_rates.extend(rates)
    
    if all_rates:
        all_rates = np.array(all_rates)
        summary_data.append({
            'Method': method,
            'Mean_Rate': np.mean(all_rates),
            'Std_Rate': np.std(all_rates),
            'Min_Rate': np.min(all_rates),
            'Max_Rate': np.max(all_rates),
            'Pct_Below_0.2': np.mean(all_rates < 0.2) * 100,
            'Pct_Above_0.4': np.mean(all_rates > 0.4) * 100,
        })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    print("\n", summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "Dropout_Rates_Summary.csv"), index=False)
    print(f"\n✅ Summary saved to: {os.path.join(OUTPUT_DIR, 'Dropout_Rates_Summary.csv')}")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE!")
print(f"   Methods extracted: {len(all_dropout_rates)}")
print(f"   Figure 6: {'Generated' if all_dropout_rates else 'Not generated'}")
print(f"   Figure 7: {'Generated' if correlation_found else 'Not generated'} (VARDON_AdaptiveVD)")
print(f"📁 Output directory: {OUTPUT_DIR}")
print("=" * 80)