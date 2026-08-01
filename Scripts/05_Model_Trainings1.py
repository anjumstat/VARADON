# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 22:25:36 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
PAPER 1: Adaptive and Real Variational Dropout for Enzyme Classification

Corrected experimental design:
1. 90/10 split is created ONCE at the start.
2. The 10% holdout test set is NOT used during model/hyperparameter comparison.
3. Stratified 10-fold CV is performed only on the 90% training+validation set.
4. StandardScaler is fitted inside each CV fold using only that fold's training data.
5. The best model/hyperparameter setting is selected using CV performance only.
6. The selected model is retrained on the full 90% training+validation set.
7. Final independent test evaluation is performed once on the untouched 10% test set.
8. AdaptiveVariationalDropout inference behavior is corrected.
9. ADDED: Ablation analysis for VARDON_Full components.
10. ADDED: Two new baseline models (MLP with BatchNorm, Residual MLP).
11. ADDED: Leave-One-Species-Out cross-validation for cross-species generalization.
12. ADDED: Checkpoint/resume functionality - if interrupted, continues from where it stopped.
13. ADDED: Test results for ALL models on the holdout test set.
"""

import os
import json
import time
import warnings
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# Optional: force CPU-only execution.
try:
    tf.config.set_visible_devices([], "GPU")
    print("✅ Running on CPU mode")
except Exception:
    print("⚠️ Could not change GPU visibility; continuing with available devices.")


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = r"D:\zebfish\revision\zebfish_processed_results\combined_data\binary_classification_dataset.csv"
BASE_DIR = r"D:\zebfish\revision\VARDON_Results_Corrected_0.01"
os.makedirs(BASE_DIR, exist_ok=True)

# Checkpoint file to track completed runs
CHECKPOINT_FILE = os.path.join(BASE_DIR, "completed_runs.json")

LEARNING_RATES = [0.01, 0.001, 0.0001]
BATCH_SIZES = [32, 64, 128]

EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
N_FOLDS = 10
TEST_SIZE = 0.10
RANDOM_STATE = 42

SELECTION_METRIC = "mean_mcc"

CLASS_NAMES = ["Non-enzyme", "Enzyme"]

# Path to species labels for LOSO
SPECIES_DATA_PATH = r"D:\zebfish\revision\zebfish_processed_results\combined_data\binary_classification_with_species.csv"


METHODS = {
    # === BASELINES ===
    "Logistic_Regression": {
        "type": "baseline",
        "variant": "true_logistic",
        "description": "Softmax Logistic Regression / Linear Softmax Classifier; no hidden layers",
    },
    "Gaussian_Dropout_NN": {
        "type": "baseline",
        "variant": "gaussian_dropout",
        "description": "Gaussian Dropout Neural Network",
    },
    "Standard_Dropout": {
        "type": "baseline",
        "variant": "standard_dropout",
        "description": "Standard Bernoulli Dropout Neural Network",
    },
    "MLP_BatchNorm": {
        "type": "baseline",
        "variant": "mlp_batchnorm",
        "description": "MLP with Batch Normalization (no dropout)",
    },
    "Residual_MLP": {
        "type": "baseline",
        "variant": "residual_mlp",
        "description": "Residual MLP with Batch Normalization",
    },

    # === PROPOSED / MODIFIED VARDON VARIANTS ===
    "VARDON_RealVD": {
        "type": "dropout_variant",
        "variant": "real_vd_only",
        "description": "Real Variational Dropout variant",
    },
    "VARDON_AdaptiveVD": {
        "type": "dropout_variant",
        "variant": "adaptive_vd_only",
        "description": "Adaptive Variational Dropout variant with learnable per-neuron rates",
    },
    "VARDON_RealVD_Adaptive": {
        "type": "dropout_variant",
        "variant": "real_adaptive",
        "description": "Combined Real Variational Dropout and Adaptive Variational Dropout",
    },
    "VARDON_Full": {
        "type": "dropout_variant",
        "variant": "full",
        "description": "Full VARDON with soft feature gate, dynamic sparsity, RealVD, and AdaptiveVD",
    },
    "VARDON_Light": {
        "type": "dropout_variant",
        "variant": "light",
        "description": "Lightweight VARDON variant for efficiency",
    },
}

# ============================================================================
# ABLATION CONFIGURATIONS (for VARDON_Full analysis)
# ============================================================================

ABLATION_CONFIGS = {
    "VARDON_Full_No_Gate": {
        "type": "ablation",
        "variant": "full_no_gate",
        "description": "VARDON_Full without Feature Importance Gate",
        "parent": "VARDON_Full",
        "removed_component": "feature_gate"
    },
    "VARDON_Full_No_Sparsity": {
        "type": "ablation",
        "variant": "full_no_sparsity",
        "description": "VARDON_Full without Dynamic Sparsity Regularizer",
        "parent": "VARDON_Full",
        "removed_component": "dynamic_sparsity"
    },
    "VARDON_Full_No_Both": {
        "type": "ablation",
        "variant": "full_no_both",
        "description": "VARDON_Full without Feature Gate and Dynamic Sparsity",
        "parent": "VARDON_Full",
        "removed_component": "both"
    },
}

# Merge ablations into METHODS
METHODS.update(ABLATION_CONFIGS)


# ============================================================================
# CHECKPOINT FUNCTIONS
# ============================================================================

def load_checkpoint():
    """Load completed runs from checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"completed_runs": []}

def save_checkpoint(completed_runs):
    """Save completed runs to checkpoint file."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({"completed_runs": completed_runs}, f, indent=2)

def is_run_completed(method_name, learning_rate, batch_size):
    """Check if a specific run has been completed."""
    checkpoint = load_checkpoint()
    run_key = f"{method_name}_lr_{learning_rate}_bs_{batch_size}"
    return run_key in checkpoint["completed_runs"]

def mark_run_completed(method_name, learning_rate, batch_size):
    """Mark a run as completed."""
    checkpoint = load_checkpoint()
    run_key = f"{method_name}_lr_{learning_rate}_bs_{batch_size}"
    if run_key not in checkpoint["completed_runs"]:
        checkpoint["completed_runs"].append(run_key)
        save_checkpoint(checkpoint["completed_runs"])

def get_pending_runs():
    """Get list of runs that need to be executed."""
    pending = []
    for method_name in METHODS.keys():
        for lr in LEARNING_RATES:
            for bs in BATCH_SIZES:
                if not is_run_completed(method_name, lr, bs):
                    pending.append((method_name, lr, bs))
    return pending


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_auc(y_true, y_score):
    """Safely compute ROC-AUC. Returns NaN if only one class is present."""
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


# ============================================================================
# CUSTOM LAYERS AND REGULARIZERS
# ============================================================================

@tf.keras.utils.register_keras_serializable(package="VARDON")
class RealVariationalDropout(layers.Layer):
    """
    Real Variational Dropout-style layer.
    Uses multiplicative Gaussian noise with learnable log-alpha parameters.
    """

    def __init__(self, units, init_drop_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.init_drop_rate = init_drop_rate
        self.eps = 1e-8

    def build(self, input_shape):
        alpha_init = self.init_drop_rate / (1.0 - self.init_drop_rate + self.eps)
        log_alpha_init = np.log(alpha_init + self.eps)

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

    def get_dropout_rates(self):
        alpha = tf.exp(self.log_alpha).numpy()
        dropout_rate = alpha / (1.0 + alpha + self.eps)
        return dropout_rate

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "init_drop_rate": self.init_drop_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="VARDON")
class AdaptiveVariationalDropout(layers.Layer):
    """
    Adaptive dropout layer with learnable per-neuron dropout rates.

    Important correction:
    Because training uses inverted dropout scaling, inference must return inputs
    directly, not inputs * (1 - drop_rate).
    """

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
        else:
            self.drop_logits = None

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
            tf.shape(inputs),
            p=1.0 - drop_rate,
            dtype=inputs.dtype,
        )
        gaussian_noise = tf.random.normal(tf.shape(inputs), dtype=inputs.dtype) * self.noise_scale
        combined_noise = bernoulli_mask * (1.0 + gaussian_noise)
        scale = 1.0 / (1.0 - drop_rate + self.eps)
        return inputs * combined_noise * scale

    def get_drop_rates(self):
        if self.learnable:
            return tf.sigmoid(self.drop_logits).numpy()
        return np.ones(self.units) * self.initial_drop_rate

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "initial_drop_rate": self.initial_drop_rate,
            "learnable": self.learnable,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="VARDON")
class DynamicSparsityRegularizer(regularizers.Regularizer):
    """Dynamic sparsity regularizer with progressive target sparsity."""

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


@tf.keras.utils.register_keras_serializable(package="VARDON")
class FeatureImportanceGate(layers.Layer):
    """
    Differentiable soft feature gate.

    This replaces the previous hard top-k gate. The previous top-k version was not
    smoothly differentiable. This soft gate learns a continuous importance weight
    for every input feature.
    """

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

    def get_gate_values(self):
        return tf.sigmoid(self.feature_importance / self.temperature).numpy()

    def get_config(self):
        config = super().get_config()
        config.update({
            "keep_ratio": self.keep_ratio,
            "temperature": self.temperature,
            "gate_regularization": self.gate_regularization,
        })
        return config


# ============================================================================
# CALLBACKS
# ============================================================================

class DynamicSparsityCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        for layer in self.model.layers:
            if hasattr(layer, "kernel_regularizer"):
                reg = layer.kernel_regularizer
                if hasattr(reg, "update_epoch"):
                    reg.update_epoch(epoch)


class MCCCallback(callbacks.Callback):
    """Calculate MCC on validation data after each epoch."""

    def __init__(self, validation_data):
        super().__init__()
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred_proba = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        logs["val_mcc"] = matthews_corrcoef(self.y_val, y_pred)


# ============================================================================
# MODEL BUILDERS
# ============================================================================

def build_softmax_logistic_regression_model(input_shape, num_classes, learning_rate):
    """Single-layer softmax classifier; equivalent to multinomial logistic regression."""
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="Softmax_Logistic_Regression")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_mlp_batchnorm_model(input_shape, num_classes, learning_rate):
    """NEW: MLP with Batch Normalization (no dropout)"""
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(512, activation="relu", name="dense1"),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu", name="dense2"),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu", name="dense3"),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu", name="dense4"),
        layers.BatchNormalization(),
        layers.Dense(64, activation="relu", name="dense5"),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="MLP_BatchNorm")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_residual_mlp_model(input_shape, num_classes, learning_rate):
    """NEW: Residual MLP with Batch Normalization"""
    inputs = layers.Input(shape=(input_shape,), name="input")
    x = inputs
    
    # Stage 1
    x = layers.Dense(512, activation="relu", name="dense1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    
    # Stage 2 with residual
    shortcut = x
    x = layers.Dense(256, activation="relu", name="dense2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = layers.Dense(256, name="shortcut1")(shortcut)
    x = layers.Add(name="residual1")([x, shortcut])
    
    # Stage 3 with residual
    shortcut = x
    x = layers.Dense(256, activation="relu", name="dense3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = layers.Dense(256, name="shortcut2")(shortcut)
    x = layers.Add(name="residual2")([x, shortcut])
    
    # Stage 4
    x = layers.Dense(128, activation="relu", name="dense4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    
    # Stage 5
    x = layers.Dense(64, activation="relu", name="dense5")(x)
    x = layers.BatchNormalization(name="bn5")(x)
    
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Residual_MLP")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_standard_dropout_model(input_shape, num_classes, learning_rate):
    """Standard Bernoulli Dropout baseline."""
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(512, activation="relu", name="dense1"),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu", name="dense2"),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu", name="dense3"),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(64, activation="relu", name="dense4"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="Standard_Dropout_NN")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_gaussian_dropout_model(input_shape, num_classes, learning_rate):
    """Gaussian Dropout Neural Network baseline."""
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(512, activation="relu", name="dense1"),
        layers.GaussianDropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu", name="dense2"),
        layers.GaussianDropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu", name="dense3"),
        layers.GaussianDropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(64, activation="relu", name="dense4"),
        layers.GaussianDropout(0.2),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="Gaussian_Dropout_NN")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_dropout_model(input_shape, num_classes, learning_rate, variant="real_vd_only"):
    """Build proposed VARDON dropout-focused variants."""
    inputs = layers.Input(shape=(input_shape,), name="input")
    x = inputs

    # Determine which components to include (for ablation)
    use_gate = variant in ["full", "full_no_sparsity"]
    use_sparsity = variant in ["full", "full_no_gate"]
    
    # For ablation: full_no_both removes both
    if variant == "full_no_both":
        use_gate = False
        use_sparsity = False
    
    if use_gate:
        x = FeatureImportanceGate(keep_ratio=0.8, name="feature_gate")(x)

    x = layers.BatchNormalization(name="bn1")(x)

    # Stage 1
    if use_sparsity:
        x = layers.Dense(
            512,
            activation="relu",
            kernel_regularizer=DynamicSparsityRegularizer(total_epochs=EPOCHS),
            name="dense1",
        )(x)
    else:
        x = layers.Dense(512, activation="relu", name="dense1")(x)

    if variant in ["real_vd_only", "real_adaptive", "full", "light", "full_no_gate", "full_no_sparsity", "full_no_both"]:
        x = RealVariationalDropout(512, init_drop_rate=0.2, name="rvd1")(x)
    if variant in ["adaptive_vd_only", "real_adaptive", "full", "light", "full_no_gate", "full_no_sparsity", "full_no_both"]:
        x = AdaptiveVariationalDropout(512, initial_drop_rate=0.2, name="avd1")(x)

    x = layers.BatchNormalization(name="bn2")(x)

    # Stage 2 with residual connection
    shortcut = x

    if use_sparsity:
        x = layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=DynamicSparsityRegularizer(total_epochs=EPOCHS),
            name="dense2",
        )(x)
    else:
        x = layers.Dense(256, activation="relu", name="dense2")(x)

    if variant in ["real_vd_only", "real_adaptive", "full", "light", "full_no_gate", "full_no_sparsity", "full_no_both"]:
        x = RealVariationalDropout(256, init_drop_rate=0.3, name="rvd2")(x)
    if variant in ["adaptive_vd_only", "real_adaptive", "full", "light", "full_no_gate", "full_no_sparsity", "full_no_both"]:
        x = AdaptiveVariationalDropout(256, initial_drop_rate=0.3, name="avd2")(x)

    x = layers.BatchNormalization(name="bn3")(x)

    if use_sparsity:
        x = layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=DynamicSparsityRegularizer(total_epochs=EPOCHS),
            name="dense3",
        )(x)
    else:
        x = layers.Dense(256, activation="relu", name="dense3")(x)

    if variant in ["real_vd_only", "real_adaptive", "full", "light", "full_no_gate", "full_no_sparsity", "full_no_both"]:
        x = RealVariationalDropout(256, init_drop_rate=0.3, name="rvd3")(x)
    if variant in ["adaptive_vd_only", "real_adaptive", "full", "light", "full_no_gate", "full_no_sparsity", "full_no_both"]:
        x = AdaptiveVariationalDropout(256, initial_drop_rate=0.3, name="avd3")(x)

    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = layers.Dense(256, name="shortcut")(shortcut)
    x = layers.Add(name="residual_add")([x, shortcut])
    x = layers.BatchNormalization(name="bn4")(x)

    # Stage 3
    x = layers.Dense(128, activation="relu", name="dense4")(x)

    if variant in ["real_vd_only", "real_adaptive", "full", "light", "full_no_gate", "full_no_sparsity", "full_no_both"]:
        x = RealVariationalDropout(128, init_drop_rate=0.4, name="rvd4")(x)
    if variant in ["adaptive_vd_only", "real_adaptive", "full", "light", "full_no_gate", "full_no_sparsity", "full_no_both"]:
        x = AdaptiveVariationalDropout(128, initial_drop_rate=0.4, name="avd4")(x)

    if variant == "light":
        x = layers.Dense(32, activation="relu", name="dense5")(x)
    else:
        x = layers.Dense(64, activation="relu", name="dense5")(x)

    x = layers.Dropout(0.3, name="final_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"VARDON_{variant}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_model(method_name, config, input_shape, num_classes, learning_rate):
    """Factory function for all model types."""
    if config["variant"] == "true_logistic":
        return build_softmax_logistic_regression_model(input_shape, num_classes, learning_rate)
    if config["variant"] == "standard_dropout":
        return build_standard_dropout_model(input_shape, num_classes, learning_rate)
    if config["variant"] == "gaussian_dropout":
        return build_gaussian_dropout_model(input_shape, num_classes, learning_rate)
    if config["variant"] == "mlp_batchnorm":
        return build_mlp_batchnorm_model(input_shape, num_classes, learning_rate)
    if config["variant"] == "residual_mlp":
        return build_residual_mlp_model(input_shape, num_classes, learning_rate)
    if config["type"] == "dropout_variant" or config["type"] == "ablation":
        return build_dropout_model(input_shape, num_classes, learning_rate, config["variant"])
    raise ValueError(f"Unknown model configuration for {method_name}: {config}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_binary_data(data_path):
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    if "Label" not in df.columns:
        raise ValueError("Dataset must contain a 'Label' column with 0/1 class labels.")

    X = df.drop(columns=["Label"]).values.astype(np.float32)
    y = df["Label"].values.astype(int)

    if set(np.unique(y)) - {0, 1}:
        raise ValueError("This script expects binary labels coded as 0 and 1.")

    print(f"Data shape: {X.shape}")
    print("Class distribution:")
    print(f"  {CLASS_NAMES[0]}: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
    print(f"  {CLASS_NAMES[1]}: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")

    return X, y


def load_species_data():
    """Load species labels for LOSO evaluation."""
    if not os.path.exists(SPECIES_DATA_PATH):
        print(f"⚠️ Species data not found: {SPECIES_DATA_PATH}")
        return None
    
    df = pd.read_csv(SPECIES_DATA_PATH)
    if "Data_Source" not in df.columns:
        print("⚠️ No 'Data_Source' column found for species")
        return None
    
    return df


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_training_history(history_dict, output_dir, method_name, fold_num):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].plot(history_dict.get("accuracy", []), label="Train Accuracy", linewidth=2)
    axes[0, 0].plot(history_dict.get("val_accuracy", []), label="Val Accuracy", linewidth=2)
    axes[0, 0].set_title("Accuracy over Epochs")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history_dict.get("loss", []), label="Train Loss", linewidth=2)
    axes[0, 1].plot(history_dict.get("val_loss", []), label="Val Loss", linewidth=2)
    axes[0, 1].set_title("Loss over Epochs")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(history_dict.get("precision", []), label="Train Precision", linewidth=2)
    axes[0, 2].plot(history_dict.get("val_precision", []), label="Val Precision", linewidth=2)
    axes[0, 2].set_title("Precision over Epochs")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Precision")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(history_dict.get("recall", []), label="Train Recall", linewidth=2)
    axes[1, 0].plot(history_dict.get("val_recall", []), label="Val Recall", linewidth=2)
    axes[1, 0].set_title("Recall over Epochs")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history_dict.get("auc", []), label="Train AUC", linewidth=2)
    axes[1, 1].plot(history_dict.get("val_auc", []), label="Val AUC", linewidth=2)
    axes[1, 1].set_title("AUC over Epochs")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("AUC")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(history_dict.get("val_mcc", []), label="Val MCC", linewidth=2)
    axes[1, 2].set_title("MCC over Epochs")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("MCC")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f"{method_name} - Fold {fold_num} Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_history_fold{fold_num}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, output_path, title, cmap="Blues", average=False):
    plt.figure(figsize=(8, 6))
    fmt = ".1f" if average else "d"
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_stability(feature_importances_list, output_dir, method_name):
    if len(feature_importances_list) < 2:
        return

    importance_array = np.array(feature_importances_list)
    mean_importance = np.mean(importance_array, axis=0)
    std_importance = np.std(importance_array, axis=0)

    top_20_idx = np.argsort(mean_importance)[-20:]
    top_20_importance = mean_importance[top_20_idx]
    top_20_std = std_importance[top_20_idx]

    plt.figure(figsize=(12, 6))
    plt.barh(range(len(top_20_importance)), top_20_importance, xerr=top_20_std, capsize=3, alpha=0.7)
    plt.yticks(range(len(top_20_importance)), [f"F{idx}" for idx in top_20_idx])
    plt.xlabel("Mean Feature Importance")
    plt.title(f"{method_name} - Top 20 Features (Mean ± Std across CV folds)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_stability.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# FEATURE STABILITY FUNCTIONS
# ============================================================================

def extract_feature_importance(model, layer_name="dense1"):
    """Extract mean absolute input weights from the first dense layer."""
    for layer in model.layers:
        if layer.name == layer_name and hasattr(layer, "get_weights"):
            weights = layer.get_weights()
            if weights:
                kernel = weights[0]
                if kernel.ndim == 2:
                    return np.mean(np.abs(kernel), axis=1)
    return None


def calculate_jaccard_stability(feature_importances_list, top_k=50):
    """Calculate Jaccard similarity of top-k features across folds."""
    if len(feature_importances_list) < 2:
        return 0.0

    top_feature_sets = []
    for importance in feature_importances_list:
        if importance is not None and len(importance) >= top_k:
            top_indices = np.argsort(importance)[-top_k:].tolist()
            top_feature_sets.append(set(top_indices))

    if len(top_feature_sets) < 2:
        return 0.0

    scores = []
    for i in range(len(top_feature_sets)):
        for j in range(i + 1, len(top_feature_sets)):
            union = len(top_feature_sets[i] | top_feature_sets[j])
            if union > 0:
                intersection = len(top_feature_sets[i] & top_feature_sets[j])
                scores.append(intersection / union)

    return float(np.mean(scores)) if scores else 0.0


# ============================================================================
# CROSS-VALIDATION EXPERIMENT (with checkpoint)
# ============================================================================

def run_cross_validation_experiment(
    method_name,
    config,
    learning_rate,
    batch_size,
    X_train_val_raw,
    y_train_val,
    train_val_indices,
):
    """
    Run stratified 10-fold CV on the training+validation set.
    """
    
    # Check if already completed
    if is_run_completed(method_name, learning_rate, batch_size):
        print(f"\n⏭️ SKIPPING: {method_name} | lr={learning_rate} | batch_size={batch_size} (already completed)")
        return None

    lr_str = f"{learning_rate:.5f}".replace(".", "_")
    output_dir = ensure_dir(os.path.join(BASE_DIR, "cv_runs", f"lr_{lr_str}_bs_{batch_size}", method_name))
    npy_dir = ensure_dir(os.path.join(output_dir, "npy_files"))
    csv_dir = ensure_dir(os.path.join(output_dir, "csv_files"))
    plots_dir = ensure_dir(os.path.join(output_dir, "plots"))
    models_dir = ensure_dir(os.path.join(output_dir, "models"))

    num_classes = 2
    input_shape = X_train_val_raw.shape[1]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_acc = []
    fold_precision = []
    fold_recall = []
    fold_f1 = []
    fold_mcc = []
    fold_auc = []
    fold_times = []
    fold_epochs = []
    all_predictions = []
    confusion_matrices = []
    feature_importances = []

    print(f"\n{'=' * 80}")
    print(f"CV RUN: {method_name} | lr={learning_rate} | batch_size={batch_size}")
    print(f"Description: {config['description']}")
    print(f"{'=' * 80}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val_raw, y_train_val), start=1):
        print(f"\nFold {fold}/{N_FOLDS} - {method_name}")

        X_tr_raw = X_train_val_raw[train_idx]
        X_val_raw = X_train_val_raw[val_idx]
        y_tr = y_train_val[train_idx]
        y_val = y_train_val[val_idx]

        # Correct scaling: fit scaler only on the training fold.
        fold_scaler = StandardScaler()
        X_tr = fold_scaler.fit_transform(X_tr_raw).astype(np.float32)
        X_val = fold_scaler.transform(X_val_raw).astype(np.float32)

        save_json({
            "train_indices_original_dataset": train_val_indices[train_idx].tolist(),
            "val_indices_original_dataset": train_val_indices[val_idx].tolist(),
        }, os.path.join(npy_dir, f"fold{fold}_indices.json"))

        save_json({
            "mean": fold_scaler.mean_.tolist(),
            "scale": fold_scaler.scale_.tolist(),
        }, os.path.join(npy_dir, f"fold{fold}_scaler_params.json"))

        y_tr_cat = tf.keras.utils.to_categorical(y_tr, num_classes)
        y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)

        tf.keras.backend.clear_session()
        model = build_model(method_name, config, input_shape, num_classes, learning_rate)

        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )
        dyn_callback = DynamicSparsityCallback()
        mcc_callback = MCCCallback(validation_data=(X_val, y_val))

        start_time = time.time()
        history = model.fit(
            X_tr,
            y_tr_cat,
            epochs=EPOCHS,
            batch_size=batch_size,
            validation_data=(X_val, y_val_cat),
            verbose=0,
            callbacks=[early_stop, dyn_callback, mcc_callback],
        )
        fold_time = time.time() - start_time
        fold_epochs.append(len(history.history.get("loss", [])))

        for metric, values in history.history.items():
            np.save(os.path.join(npy_dir, f"fold{fold}_{metric}.npy"), np.array(values))

        plot_training_history(history.history, plots_dir, method_name, fold)

        y_pred = model.predict(X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_pred_proba = y_pred[:, 1]

        np.save(os.path.join(npy_dir, f"fold{fold}_predictions.npy"), y_pred)
        np.save(os.path.join(npy_dir, f"fold{fold}_predicted_classes.npy"), y_pred_classes)
        np.save(os.path.join(npy_dir, f"fold{fold}_predicted_proba.npy"), y_pred_proba)
        np.save(os.path.join(npy_dir, f"fold{fold}_true_labels.npy"), y_val)

        acc = accuracy_score(y_val, y_pred_classes)
        precision = precision_score(y_val, y_pred_classes, zero_division=0)
        recall = recall_score(y_val, y_pred_classes, zero_division=0)
        f1 = f1_score(y_val, y_pred_classes, zero_division=0)
        mcc = matthews_corrcoef(y_val, y_pred_classes)
        auc = safe_auc(y_val, y_pred_proba)

        cm = confusion_matrix(y_val, y_pred_classes)
        confusion_matrices.append(cm)
        np.save(os.path.join(npy_dir, f"fold{fold}_confusion_matrix.npy"), cm)
        plot_confusion_matrix(
            cm,
            os.path.join(plots_dir, f"confusion_matrix_fold{fold}.png"),
            f"{method_name} - Fold {fold} Confusion Matrix",
            cmap="Blues",
        )

        importance = extract_feature_importance(model, layer_name="dense1")
        if importance is not None:
            feature_importances.append(importance)
            np.save(os.path.join(npy_dir, f"fold{fold}_feature_importance.npy"), importance)

        for local_i, (true_label, pred_label, proba) in enumerate(zip(y_val, y_pred_classes, y_pred_proba)):
            original_sample_idx = int(train_val_indices[val_idx[local_i]])
            all_predictions.append({
                "fold": fold,
                "sample_idx_original_dataset": original_sample_idx,
                "true_label": int(true_label),
                "predicted_label": int(pred_label),
                "predicted_proba_enzyme": float(proba),
                "correct": bool(true_label == pred_label),
            })

        print(
            f"  Acc: {acc:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, "
            f"F1: {f1:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}, Epochs: {fold_epochs[-1]}"
        )

        fold_acc.append(acc)
        fold_precision.append(precision)
        fold_recall.append(recall)
        fold_f1.append(f1)
        fold_mcc.append(mcc)
        fold_auc.append(auc)
        fold_times.append(fold_time)

        model.save(os.path.join(models_dir, f"fold{fold}_model.keras"))

    jaccard_stability = calculate_jaccard_stability(feature_importances, top_k=50)
    if len(feature_importances) >= 2:
        plot_feature_stability(feature_importances, plots_dir, method_name)

    avg_cm = np.mean(confusion_matrices, axis=0)
    np.save(os.path.join(npy_dir, "average_confusion_matrix.npy"), avg_cm)
    plot_confusion_matrix(
        avg_cm,
        os.path.join(plots_dir, "average_confusion_matrix.png"),
        f"{method_name} - Average Confusion Matrix ({N_FOLDS}-Fold CV)",
        cmap="Blues",
        average=True,
    )

    fold_metrics_df = pd.DataFrame({
        "Fold": range(1, N_FOLDS + 1),
        "Accuracy": fold_acc,
        "Precision": fold_precision,
        "Recall": fold_recall,
        "F1": fold_f1,
        "MCC": fold_mcc,
        "AUC": fold_auc,
        "Training_Time_Seconds": fold_times,
        "Epochs_Trained": fold_epochs,
    })
    fold_metrics_df.to_csv(os.path.join(csv_dir, "Fold_Metrics.csv"), index=False)

    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv(os.path.join(csv_dir, "All_CV_Predictions.csv"), index=False)

    recommended_epochs = int(max(1, round(np.median(fold_epochs))))

    summary = {
        "status": "success",
        "method": method_name,
        "description": config["description"],
        "variant": config["variant"],
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "mean_accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc)),
        "mean_precision": float(np.mean(fold_precision)),
        "std_precision": float(np.std(fold_precision)),
        "mean_recall": float(np.mean(fold_recall)),
        "std_recall": float(np.std(fold_recall)),
        "mean_f1": float(np.mean(fold_f1)),
        "std_f1": float(np.std(fold_f1)),
        "mean_mcc": float(np.mean(fold_mcc)),
        "std_mcc": float(np.std(fold_mcc)),
        "mean_auc": float(np.nanmean(fold_auc)),
        "std_auc": float(np.nanstd(fold_auc)),
        "feature_stability_jaccard": float(jaccard_stability),
        "mean_training_time_seconds": float(np.mean(fold_times)),
        "recommended_epochs_for_final_training": recommended_epochs,
        "output_dir": output_dir,
        "fold_accuracy_values": [float(x) for x in fold_acc],
        "fold_mcc_values": [float(x) for x in fold_mcc],
        "fold_f1_values": [float(x) for x in fold_f1],
        "fold_auc_values": [float(x) for x in fold_auc],
    }

    pd.DataFrame([
        {k: v for k, v in summary.items() if not isinstance(v, list)}
    ]).to_csv(os.path.join(csv_dir, "Experiment_Summary.csv"), index=False)

    print(f"\n{method_name} CV Summary")
    print("-" * 60)
    print(f"Mean Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"Mean F1:       {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    print(f"Mean MCC:      {summary['mean_mcc']:.4f} ± {summary['std_mcc']:.4f}")
    print(f"Mean AUC:      {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
    print(f"Jaccard Stability: {summary['feature_stability_jaccard']:.4f}")
    print(f"Recommended final training epochs: {recommended_epochs}")

    # Mark as completed
    mark_run_completed(method_name, learning_rate, batch_size)

    return summary


# ============================================================================
# ALL MODELS TEST EVALUATION (NEW)
# ============================================================================

def run_all_models_test_evaluation(results_list, X_train_val_raw, y_train_val, X_test_raw, y_test, test_indices):
    """
    Evaluate ALL successful models on the holdout test set.
    
    This is different from the previous code which only evaluated the best model.
    Now every model/hyperparameter configuration gets evaluated on the test set.
    
    Output: Paper1_Test_Results_All_Models.csv
    """
    successful = [r for r in results_list if r.get("status") == "success"]
    
    if not successful:
        print("No successful CV results available for all-model test evaluation.")
        return pd.DataFrame()
    
    print("\n" + "=" * 80)
    print("INDEPENDENT TEST EVALUATION FOR ALL MODELS")
    print("=" * 80)
    print(f"Evaluating {len(successful)} model configurations on test set...")
    print("-" * 80)
    
    all_test_results = []
    
    for idx, r in enumerate(successful):
        method_name = r["method"]
        config = METHODS[method_name]
        learning_rate = r["learning_rate"]
        batch_size = r["batch_size"]
        final_epochs = r["recommended_epochs_for_final_training"]
        
        lr_str = f"{learning_rate:.5f}".replace(".", "_")
        setting_name = f"{method_name}_lr_{lr_str}_bs_{batch_size}"
        
        print(f"\n[{idx+1}/{len(successful)}] Testing: {setting_name}")
        
        # Scale data using full training+validation set
        final_scaler = StandardScaler()
        X_train_val = final_scaler.fit_transform(X_train_val_raw).astype(np.float32)
        X_test = final_scaler.transform(X_test_raw).astype(np.float32)
        
        num_classes = 2
        input_shape = X_train_val.shape[1]
        y_train_val_cat = tf.keras.utils.to_categorical(y_train_val, num_classes)
        
        # Build and train model
        tf.keras.backend.clear_session()
        model = build_model(method_name, config, input_shape, num_classes, learning_rate)
        
        # Train with early stopping (using validation split)
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=0
        )
        
        start_time = time.time()
        history = model.fit(
            X_train_val, y_train_val_cat,
            epochs=final_epochs,
            batch_size=batch_size,
            validation_split=0.1,  # Use 10% of training as validation
            verbose=0,
            callbacks=[early_stop]
        )
        training_time = time.time() - start_time
        
        # Evaluate on test set
        y_test_pred = model.predict(X_test, verbose=0)
        y_test_classes = np.argmax(y_test_pred, axis=1)
        y_test_proba = y_test_pred[:, 1]
        
        # Compute all metrics
        test_acc = accuracy_score(y_test, y_test_classes)
        test_precision = precision_score(y_test, y_test_classes, zero_division=0)
        test_recall = recall_score(y_test, y_test_classes, zero_division=0)
        test_f1 = f1_score(y_test, y_test_classes, zero_division=0)
        test_mcc = matthews_corrcoef(y_test, y_test_classes)
        test_auc = safe_auc(y_test, y_test_proba)
        
        # Confusion matrix
        test_cm = confusion_matrix(y_test, y_test_classes)
        
        # Store results
        result_row = {
            "Method": method_name,
            "Description": r["description"],
            "Learning_Rate": learning_rate,
            "Batch_Size": batch_size,
            "Final_Training_Epochs": final_epochs,
            "Final_Training_Time_Seconds": training_time,
            "CV_Mean_Accuracy": r["mean_accuracy"],
            "CV_Std_Accuracy": r["std_accuracy"],
            "CV_Mean_Precision": r["mean_precision"],
            "CV_Std_Precision": r["std_precision"],
            "CV_Mean_Recall": r["mean_recall"],
            "CV_Std_Recall": r["std_recall"],
            "CV_Mean_F1": r["mean_f1"],
            "CV_Std_F1": r["std_f1"],
            "CV_Mean_MCC": r["mean_mcc"],
            "CV_Std_MCC": r["std_mcc"],
            "CV_Mean_AUC": r["mean_auc"],
            "CV_Std_AUC": r["std_auc"],
            "Feature_Stability_Jaccard": r["feature_stability_jaccard"],
            "Test_Accuracy": test_acc,
            "Test_Precision": test_precision,
            "Test_Recall": test_recall,
            "Test_F1": test_f1,
            "Test_MCC": test_mcc,
            "Test_AUC": test_auc,
            "Confusion_Matrix": str(test_cm.tolist()),  # For reference
        }
        
        all_test_results.append(result_row)
        
        # Print summary
        print(f"  Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}, MCC: {test_mcc:.4f}, AUC: {test_auc:.4f}")
    
    # Create DataFrame and save
    all_test_df = pd.DataFrame(all_test_results)
    
    # Sort by Test MCC (descending)
    all_test_df = all_test_df.sort_values("Test_MCC", ascending=False)
    
    # Save to CSV
    test_results_path = os.path.join(BASE_DIR, "Paper1_Test_Results_All_Models.csv")
    all_test_df.to_csv(test_results_path, index=False)
    
    print("\n" + "=" * 80)
    print("ALL MODEL TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total models evaluated: {len(all_test_df)}")
    print(f"Results saved to: {test_results_path}")
    
    # Print top 10 models by Test MCC
    print("\n🏆 TOP 10 MODELS BY TEST MCC:")
    print("-" * 80)
    top_df = all_test_df[["Method", "Learning_Rate", "Batch_Size", "Test_Accuracy", "Test_F1", "Test_MCC", "Test_AUC"]].head(10)
    print(top_df.to_string(index=False))
    
    return all_test_df


# ============================================================================
# LEAVE-ONE-SPECIES-OUT EVALUATION
# ============================================================================

# ============================================================================
# LEAVE-ONE-SPECIES-OUT EVALUATION (UPDATED: ALL MODELS)
# ============================================================================

# ============================================================================
# LEAVE-ONE-SPECIES-OUT EVALUATION (UPDATED: ALL MODELS + PRECISION/RECALL)
# ============================================================================

def run_loso_evaluation():
    """
    Run Leave-One-Species-Out cross-validation for ALL models.
    This evaluates cross-species generalization for every method.
    
    UPDATED: Now runs for ALL 13 methods and saves Precision/Recall.
    """
    print("\n" + "=" * 80)
    print("LEAVE-ONE-SPECIES-OUT (LOSO) EVALUATION - ALL MODELS")
    print("=" * 80)
    
    # Load species data
    species_df = load_species_data()
    if species_df is None:
        print("⚠️ Skipping LOSO evaluation - species data not available")
        return None
    
    # Extract features, labels, and species
    emb_cols = [col for col in species_df.columns if col.startswith('Embedding_')]
    X = species_df[emb_cols].values.astype(np.float32)
    y = species_df['Label'].values.astype(int)
    species = species_df['Data_Source'].values
    
    unique_species = np.unique(species)
    print(f"\n🐟 Species for LOSO: {unique_species}")
    
    # ============================================================
    # UPDATED: Run ALL methods (not just 2)
    # ============================================================
    
    # Get best configuration for each method from CV results
    # Load CV summary to find best config per method
    cv_summary_path = os.path.join(BASE_DIR, "Paper1_CV_Summary_Results.csv")
    if os.path.exists(cv_summary_path):
        cv_summary = pd.read_csv(cv_summary_path)
        # For each method, find the config with highest mean_mcc
        best_configs = {}
        for method in METHODS.keys():
            method_df = cv_summary[cv_summary['method'] == method]
            if not method_df.empty:
                best_row = method_df.loc[method_df['mean_mcc'].idxmax()]
                best_configs[method] = {
                    'lr': best_row['learning_rate'],
                    'bs': best_row['batch_size'],
                    'mean_mcc': best_row['mean_mcc']
                }
                print(f"  {method}: LR={best_row['learning_rate']}, BS={best_row['batch_size']}, CV_MCC={best_row['mean_mcc']:.4f}")
    else:
        # Fallback: use default best configs
        print("⚠️ CV summary not found. Using default configurations.")
        best_configs = {
            'VARDON_Light': {'lr': 0.0001, 'bs': 32},
            'Gaussian_Dropout_NN': {'lr': 0.0001, 'bs': 32},
            'Residual_MLP': {'lr': 0.0001, 'bs': 32},
            'VARDON_Full_No_Gate': {'lr': 0.01, 'bs': 32},
            'VARDON_AdaptiveVD': {'lr': 0.0001, 'bs': 128},
            'Logistic_Regression': {'lr': 0.001, 'bs': 64},
            'Standard_Dropout': {'lr': 0.0001, 'bs': 32},
            'MLP_BatchNorm': {'lr': 0.001, 'bs': 32},
            'VARDON_RealVD': {'lr': 0.0001, 'bs': 64},
            'VARDON_RealVD_Adaptive': {'lr': 0.0001, 'bs': 128},
            'VARDON_Full': {'lr': 0.01, 'bs': 128},
            'VARDON_Full_No_Sparsity': {'lr': 0.001, 'bs': 64},
            'VARDON_Full_No_Both': {'lr': 0.001, 'bs': 64},
        }
    
    # Track which methods are completed (for checkpoint/resume)
    loso_checkpoint_file = os.path.join(BASE_DIR, "loso_checkpoint.json")
    
    def load_loso_checkpoint():
        if os.path.exists(loso_checkpoint_file):
            with open(loso_checkpoint_file, 'r') as f:
                return json.load(f)
        return {"completed": []}
    
    def save_loso_checkpoint(completed):
        with open(loso_checkpoint_file, 'w') as f:
            json.dump({"completed": completed}, f, indent=2)
    
    completed_loso = load_loso_checkpoint()["completed"]
    
    # Results container
    all_loso_results = []
    
    # Loop through each species
    for held_out_species in unique_species:
        print(f"\n{'='*60}")
        print(f"HELD-OUT SPECIES: {held_out_species.upper()}")
        print(f"{'='*60}")
        
        # Split by species
        test_mask = (species == held_out_species)
        train_mask = ~test_mask
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        print(f"  Train: {len(X_train)} samples ({len(unique_species)-1} species)")
        print(f"  Test:  {len(X_test)} samples ({held_out_species})")
        print(f"  Train enzymes: {sum(y_train)}, Test enzymes: {sum(y_test)}")
        
        # Scale data (once per species, reused for all methods)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        species_result = {
            'Held_Out_Species': held_out_species, 
            'Test_Size': len(X_test)
        }
        
        # Run ALL methods for this species
        for method_name, config in METHODS.items():
            # Check if this method is already completed for this species
            checkpoint_key = f"{held_out_species}_{method_name}"
            if checkpoint_key in completed_loso:
                print(f"\n  ⏭️ SKIPPING: {method_name} (already completed for {held_out_species})")
                continue
            
            print(f"\n  🔬 Method: {method_name}")
            
            # Get best config for this method
            if method_name in best_configs:
                lr = best_configs[method_name]['lr']
                bs = best_configs[method_name]['bs']
            else:
                # Fallback
                lr = 0.001
                bs = 32
            
            tf.keras.backend.clear_session()
            model = build_model(method_name, config, X_train.shape[1], 2, lr)
            
            y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
            
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=0
            )
            
            # Use 10% of training as validation
            history = model.fit(
                X_train_scaled, y_train_cat,
                epochs=EPOCHS,
                batch_size=bs,
                validation_split=0.1,
                verbose=0,
                callbacks=[early_stop]
            )
            
            # Evaluate
            y_pred = model.predict(X_test_scaled, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_pred_proba = y_pred[:, 1]
            
            # Compute all metrics
            acc = accuracy_score(y_test, y_pred_classes)
            precision = precision_score(y_test, y_pred_classes, zero_division=0)
            recall = recall_score(y_test, y_pred_classes, zero_division=0)
            f1 = f1_score(y_test, y_pred_classes, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred_classes)
            auc = safe_auc(y_test, y_pred_proba)
            
            # Store results - NOW INCLUDES PRECISION AND RECALL
            species_result[f'{method_name}_Accuracy'] = acc
            species_result[f'{method_name}_Precision'] = precision
            species_result[f'{method_name}_Recall'] = recall
            species_result[f'{method_name}_F1'] = f1
            species_result[f'{method_name}_MCC'] = mcc
            species_result[f'{method_name}_AUC'] = auc
            
            print(f"    Acc={acc:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}, MCC={mcc:.4f}, AUC={auc:.4f}")
            
            # Mark as completed (checkpoint)
            completed_loso.append(checkpoint_key)
            save_loso_checkpoint(completed_loso)
        
        all_loso_results.append(species_result)
    
    # Save LOSO results
    loso_df = pd.DataFrame(all_loso_results)
    loso_path = os.path.join(BASE_DIR, "LOSO_Results_All_Models.csv")
    loso_df.to_csv(loso_path, index=False)
    print(f"\n✅ LOSO results (ALL models) saved to: {loso_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("LOSO SUMMARY - ALL MODELS")
    print("=" * 80)
    
    # Compute mean performance across species for each method
    method_names = list(METHODS.keys())
    summary_data = []
    
    for method in method_names:
        acc_col = f'{method}_Accuracy'
        prec_col = f'{method}_Precision'
        rec_col = f'{method}_Recall'
        f1_col = f'{method}_F1'
        mcc_col = f'{method}_MCC'
        auc_col = f'{method}_AUC'
        
        if acc_col in loso_df.columns:
            summary_data.append({
                'Method': method,
                'Mean_Accuracy': loso_df[acc_col].mean(),
                'Std_Accuracy': loso_df[acc_col].std(),
                'Mean_Precision': loso_df[prec_col].mean(),
                'Std_Precision': loso_df[prec_col].std(),
                'Mean_Recall': loso_df[rec_col].mean(),
                'Std_Recall': loso_df[rec_col].std(),
                'Mean_F1': loso_df[f1_col].mean(),
                'Std_F1': loso_df[f1_col].std(),
                'Mean_MCC': loso_df[mcc_col].mean(),
                'Std_MCC': loso_df[mcc_col].std(),
                'Mean_AUC': loso_df[auc_col].mean(),
                'Std_AUC': loso_df[auc_col].std(),
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Mean_MCC', ascending=False)
    
    print("\n📊 Performance by Method (averaged across all 12 species):")
    print("-" * 80)
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_path = os.path.join(BASE_DIR, "LOSO_Summary_All_Models.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ LOSO summary saved to: {summary_path}")
    
    return loso_df, summary_df


# ============================================================================
# FINAL INDEPENDENT TEST EVALUATION (Best Model Only)
# ============================================================================

def run_final_test_evaluation(best_result, X_train_val_raw, y_train_val, X_test_raw, y_test, test_indices):
    """
    Train the CV-selected model on the full 90% training+validation data and
    evaluate once on the untouched 10% holdout test set.
    """
    method_name = best_result["method"]
    config = METHODS[method_name]
    learning_rate = best_result["learning_rate"]
    batch_size = best_result["batch_size"]
    final_epochs = best_result["recommended_epochs_for_final_training"]

    output_dir = ensure_dir(os.path.join(BASE_DIR, "final_selected_model_test", method_name))
    npy_dir = ensure_dir(os.path.join(output_dir, "npy_files"))
    csv_dir = ensure_dir(os.path.join(output_dir, "csv_files"))
    plots_dir = ensure_dir(os.path.join(output_dir, "plots"))
    models_dir = ensure_dir(os.path.join(output_dir, "models"))

    print("\n" + "=" * 80)
    print("FINAL INDEPENDENT TEST EVALUATION (BEST MODEL ONLY)")
    print("=" * 80)
    print(f"Selected model: {method_name}")
    print(f"Description: {best_result['description']}")
    print(f"Selected by CV metric: {SELECTION_METRIC}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Final training epochs from CV median: {final_epochs}")
    print("=" * 80)

    final_scaler = StandardScaler()
    X_train_val = final_scaler.fit_transform(X_train_val_raw).astype(np.float32)
    X_test = final_scaler.transform(X_test_raw).astype(np.float32)

    save_json({
        "mean": final_scaler.mean_.tolist(),
        "scale": final_scaler.scale_.tolist(),
    }, os.path.join(output_dir, "final_scaler_params.json"))

    num_classes = 2
    input_shape = X_train_val.shape[1]

    y_train_val_cat = tf.keras.utils.to_categorical(y_train_val, num_classes)

    tf.keras.backend.clear_session()
    final_model = build_model(method_name, config, input_shape, num_classes, learning_rate)

    start_time = time.time()
    final_history = final_model.fit(
        X_train_val,
        y_train_val_cat,
        epochs=final_epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[DynamicSparsityCallback()],
    )
    final_training_time = time.time() - start_time

    for metric, values in final_history.history.items():
        np.save(os.path.join(npy_dir, f"final_train_{metric}.npy"), np.array(values))

    y_test_pred = final_model.predict(X_test, verbose=0)
    y_test_classes = np.argmax(y_test_pred, axis=1)
    y_test_proba = y_test_pred[:, 1]

    test_acc = accuracy_score(y_test, y_test_classes)
    test_precision = precision_score(y_test, y_test_classes, zero_division=0)
    test_recall = recall_score(y_test, y_test_classes, zero_division=0)
    test_f1 = f1_score(y_test, y_test_classes, zero_division=0)
    test_mcc = matthews_corrcoef(y_test, y_test_classes)
    test_auc = safe_auc(y_test, y_test_proba)

    test_cm = confusion_matrix(y_test, y_test_classes)

    np.save(os.path.join(npy_dir, "test_predictions.npy"), y_test_pred)
    np.save(os.path.join(npy_dir, "test_predicted_classes.npy"), y_test_classes)
    np.save(os.path.join(npy_dir, "test_predicted_proba.npy"), y_test_proba)
    np.save(os.path.join(npy_dir, "test_true_labels.npy"), y_test)
    np.save(os.path.join(npy_dir, "test_confusion_matrix.npy"), test_cm)

    plot_confusion_matrix(
        test_cm,
        os.path.join(plots_dir, "test_confusion_matrix.png"),
        f"{method_name} - Independent Test Set Confusion Matrix",
        cmap="Greens",
    )

    test_predictions_df = pd.DataFrame({
        "sample_idx_original_dataset": test_indices,
        "true_label": y_test,
        "predicted_label": y_test_classes,
        "predicted_proba_enzyme": y_test_proba,
        "correct": y_test == y_test_classes,
    })
    test_predictions_df.to_csv(os.path.join(csv_dir, "Independent_Test_Predictions.csv"), index=False)

    final_result = {
        "Selected_Method": method_name,
        "Description": best_result["description"],
        "Selection_Metric": SELECTION_METRIC,
        "CV_Selected_Score": best_result[SELECTION_METRIC],
        "Learning_Rate": learning_rate,
        "Batch_Size": batch_size,
        "Final_Training_Epochs": final_epochs,
        "Final_Training_Time_Seconds": final_training_time,
        "Test_Accuracy": test_acc,
        "Test_Precision": test_precision,
        "Test_Recall": test_recall,
        "Test_F1": test_f1,
        "Test_MCC": test_mcc,
        "Test_AUC": test_auc,
    }

    pd.DataFrame([final_result]).to_csv(
        os.path.join(csv_dir, "Final_Independent_Test_Result.csv"),
        index=False,
    )
    pd.DataFrame([final_result]).to_csv(
        os.path.join(BASE_DIR, "Paper1_Final_Independent_Test_Result.csv"),
        index=False,
    )

    combined_cv_test_result = {
        "Selected_Method": method_name,
        "Description": best_result["description"],
        "Selection_Metric": SELECTION_METRIC,
        "CV_Selected_Score": best_result[SELECTION_METRIC],
        "CV_Mean_Accuracy": best_result["mean_accuracy"],
        "CV_Std_Accuracy": best_result["std_accuracy"],
        "CV_Mean_Precision": best_result["mean_precision"],
        "CV_Std_Precision": best_result["std_precision"],
        "CV_Mean_Recall": best_result["mean_recall"],
        "CV_Std_Recall": best_result["std_recall"],
        "CV_Mean_F1": best_result["mean_f1"],
        "CV_Std_F1": best_result["std_f1"],
        "CV_Mean_MCC": best_result["mean_mcc"],
        "CV_Std_MCC": best_result["std_mcc"],
        "CV_Mean_AUC": best_result["mean_auc"],
        "CV_Std_AUC": best_result["std_auc"],
        "Feature_Stability_Jaccard": best_result["feature_stability_jaccard"],
        "Learning_Rate": learning_rate,
        "Batch_Size": batch_size,
        "Final_Training_Epochs": final_epochs,
        "Test_Accuracy": test_acc,
        "Test_Precision": test_precision,
        "Test_Recall": test_recall,
        "Test_F1": test_f1,
        "Test_MCC": test_mcc,
        "Test_AUC": test_auc,
    }
    pd.DataFrame([combined_cv_test_result]).to_csv(
        os.path.join(csv_dir, "Selected_Model_CV_and_Test_Result.csv"),
        index=False,
    )
    pd.DataFrame([combined_cv_test_result]).to_csv(
        os.path.join(BASE_DIR, "Paper1_Selected_Model_CV_and_Test_Result.csv"),
        index=False,
    )

    final_model.save(os.path.join(models_dir, "final_selected_model.keras"))

    print("\nIndependent Test Results (Best Model)")
    print("-" * 60)
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1:        {test_f1:.4f}")
    print(f"Test MCC:       {test_mcc:.4f}")
    print(f"Test AUC:       {test_auc:.4f}")

    return final_result


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def run_statistical_analysis(results_list, output_dir, metric="fold_mcc_values"):
    """Run statistical comparison across CV fold scores."""
    from scipy.stats import friedmanchisquare, wilcoxon

    successful = [r for r in results_list if r.get("status") == "success" and metric in r]
    if len(successful) < 2:
        print("\nInsufficient successful runs for statistical analysis.")
        return None

    method_labels = [f"{r['method']}_lr{r['learning_rate']}_bs{r['batch_size']}" for r in successful]
    score_arrays = [np.array(r[metric], dtype=float) for r in successful]

    min_len = min(len(arr) for arr in score_arrays)
    if min_len < 2:
        print("\nInsufficient fold values for statistical analysis.")
        return None

    score_arrays = [arr[:min_len] for arr in score_arrays]

    print("\n" + "=" * 80)
    print(f"STATISTICAL ANALYSIS USING {metric}")
    print("=" * 80)

    stat_rows = []

    if len(score_arrays) >= 3:
        friedman_stat, friedman_p = friedmanchisquare(*score_arrays)
        print(f"Friedman test: statistic={friedman_stat:.4f}, p-value={friedman_p:.6f}")
        stat_rows.append({
            "Test": "Friedman",
            "Comparison": "All models/settings",
            "Statistic": friedman_stat,
            "P_Value": friedman_p,
        })
    else:
        friedman_stat, friedman_p = np.nan, np.nan

    means = [np.nanmean(arr) for arr in score_arrays]
    best_idx = int(np.nanargmax(means))
    best_label = method_labels[best_idx]
    best_scores = score_arrays[best_idx]

    print(f"Best CV setting for statistical comparison: {best_label} ({np.nanmean(best_scores):.4f})")

    p_values = []
    pair_rows = []
    for i, (label, scores) in enumerate(zip(method_labels, score_arrays)):
        if i == best_idx:
            continue
        try:
            w_stat, p_val = wilcoxon(best_scores, scores, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            w_stat, p_val = np.nan, np.nan

        p_values.append(p_val)
        pair_rows.append({
            "Test": "Wilcoxon signed-rank",
            "Comparison": f"{best_label} vs {label}",
            "Statistic": w_stat,
            "P_Value": p_val,
            "Best_Mean": float(np.nanmean(best_scores)),
            "Other_Mean": float(np.nanmean(scores)),
        })

    valid_indices = [i for i, p in enumerate(p_values) if not np.isnan(p)]
    m = len(valid_indices)
    sorted_valid = sorted(valid_indices, key=lambda i: p_values[i])
    adjusted = [np.nan] * len(p_values)
    for rank, idx in enumerate(sorted_valid):
        adjusted[idx] = min(1.0, p_values[idx] * (m - rank))

    for row, adj_p in zip(pair_rows, adjusted):
        row["Holm_Adjusted_P_Value"] = adj_p
        print(f"{row['Comparison']}: p={row['P_Value']:.6f}, Holm-adjusted p={adj_p:.6f}")

    stat_rows.extend(pair_rows)
    stat_df = pd.DataFrame(stat_rows)
    stat_path = os.path.join(output_dir, f"Statistical_Analysis_{metric}.csv")
    stat_df.to_csv(stat_path, index=False)
    print(f"Statistical analysis saved to: {stat_path}")

    return stat_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PAPER 1: ADAPTIVE AND REAL VARIATIONAL DROPOUT")
    print("RUNNING ONLY LOSO EVALUATION (ALL MODELS)")
    print("=" * 80)
    
    # ============================================================
    # SKIP ALL CV - Just run LOSO
    # ============================================================
    
    # Load data for LOSO
    species_df = load_species_data()
    if species_df is None:
        print("❌ Species data not found. Exiting.")
    else:
        # Run only LOSO
        loso_results, loso_summary = run_loso_evaluation()
        
        print("\n" + "=" * 80)
        print("✅ LOSO EVALUATION COMPLETE!")
        print("=" * 80)
        print(f"Results saved to: {os.path.join(BASE_DIR, 'LOSO_Results_All_Models.csv')}")
        print(f"Summary saved to: {os.path.join(BASE_DIR, 'LOSO_Summary_All_Models.csv')}")
        print("=" * 80)