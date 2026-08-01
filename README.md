# VARDON: Enzyme Classification Pipeline

This repository contains Python scripts for VARDON (Variational Adaptive Real Dropout Neural Network), a pipeline designed for enzyme classification across 12 fish species.

## Overview

The pipeline processes UniProt protein embeddings, performs enzyme/non-enzyme classification, and evaluates multiple VARDON variants against baseline models using cross-validation and leave-one-species-out (LOSO) evaluation strategies.

## Repository Structure
├── Data Processing/
│ ├── 01_Organize_Fish_Data.py # Organize raw data
│ ├── 02_uni_prot_data_processing.py # Process embeddings & annotations
│ ├── 03_binary_data_set_with_species.py # Create binary dataset with species
│ └── 04_Leave_one_species_out.py # Create LOSO clean splits
│
├── Model Training/
│ └── 05_Model_Trainings1.py # Main training script (all models)
│
├── Results Processing/
│ ├── 06_combine_files.py # Combine CV results with fold data
│ ├── 07_Statistical_Analysis.py # Friedman's ANOVA
│ └── 08_ROC_curves.py # Generate ROC curves
│
└── Figure Generation/
├── 09_paper_Figures.py # Publication figures (hyperparameters, training curves, boxplots)
└── 10_Figure_6_and_7.py # Dropout rates analysis

## File Descriptions

### Data Processing Scripts

#### `01_Organize_Fish_Data.py`
**Purpose**: Organize raw UniProt downloads into structured format.

**Functionality**:
- Scans raw data folder with species subfolders
- Identifies HDF5 embedding files and TSV annotation files
- Copies HDF5 files as `embeddings.h5` and TSV files as `{species}_annotations.tsv`
- Creates `metadata.csv` and `master_summary.csv`

**Input**: `D:\zebfish\`  
**Output**: `D:\zebfish_organized\`

---

#### `02_uni_prot_data_processing.py`
**Purpose**: Process embeddings and annotations to create labeled datasets with uncertainty filtering.

**Functionality**:
- Loads HDF5 embeddings and TSV annotations
- **Filters uncertain annotations** (Probable, Putative, Uncharacterized, Unknown, Predicted, Possible, Likely)
- Labels proteins as Enzyme or Non-enzyme
- Merges embeddings with labels by UniProt ID
- Saves clean dataset for deep learning
- Tracks uncertainty statistics

**Input**: `D:\zebfish_organized\`  
**Output**: `D:\zebfish_processed_results\`

---

#### `03_binary_data_set_with_species.py`
**Purpose**: Convert labeled dataset into binary format with species preservation.

**Functionality**:
- Removes metadata columns (except `Data_Source` for species)
- Converts labels to numeric: `Enzyme → 1`, `Non-enzyme → 0`
- Renames classification column to `Label`
- Saves **two versions**: with and without species

**Input**: `clean_fish_dataset_for_dl.csv`  
**Output**: 
- `binary_classification_dataset.csv` (without species)
- `binary_classification_with_species.csv` (with species for LOSO)

---

#### `04_Leave_one_species_out.py`
**Purpose**: Create clean splits for leave-one-species-out (LOSO) cross-validation.

**Functionality**:
- Creates separate train/test splits for each species
- Each species becomes the test set; all others form the training set
- Outputs clean CSV files with only embeddings and labels
- Generates summary statistics for each split

**Input**: `binary_classification_with_species.csv`  
**Output**: 
- `clean_splits/loso/{species}/train.csv`
- `clean_splits/loso/{species}/test.csv`
- `clean_splits/loso_summary.csv`
- `clean_splits/standard_splits/` (standard 70/10/20 split)

---

### Model Training Script

#### `05_Model_Trainings1.py`
**Purpose**: Run all VARDON variants and baseline models with comprehensive evaluation.

**Models Implemented**:

| Type | Methods |
|------|---------|
| **Baselines** | Logistic Regression, Gaussian Dropout NN, Standard Dropout, MLP_BatchNorm, Residual_MLP |
| **VARDON Variants** | VARDON_RealVD, VARDON_AdaptiveVD, VARDON_RealVD_Adaptive, VARDON_Full, VARDON_Light |
| **Ablations** | VARDON_Full_No_Gate, VARDON_Full_No_Sparsity, VARDON_Full_No_Both |

**Functionality**:
- Creates 90/10 train-test split (once)
- Runs 10-fold cross-validation on training set
- Performs **Leave-One-Species-Out (LOSO) evaluation** for cross-species generalization
- Evaluates all models on independent test set
- Saves training histories, metrics, and trained models
- **Checkpoint/resume capability** - continues from where it stopped
- Saves **complete test results** for all models

**Input**: `binary_classification_dataset.csv`, `binary_classification_with_species.csv`  
**Output**: 
- `cv_runs/lr_{lr}_bs_{bs}/{method}/` (CV results per configuration)
- `Paper1_CV_Summary_Results.csv`
- `Paper1_Test_Results_All_Models.csv`
- `LOSO_Results_All_Models.csv`
- `LOSO_Summary_All_Models.csv`

**Key Features**:
- All 13 methods evaluated on test set (not just best model)
- All 13 methods evaluated with LOSO
- Checkpoint system prevents duplicate runs
- Exhaustive hyperparameter search: LR ∈ {0.01, 0.001, 0.0001}, BS ∈ {32, 64, 128}

---

### Results Processing Scripts

#### `06_combine_files.py`
**Purpose**: Combine results across all methods, learning rates, and batch sizes.

**Functionality**:
- Scans result directories for `Fold_Metrics.csv`
- Parses learning rate and batch size from folder names
- Merges CV summary with fold-level data
- Creates unified summary tables
- **Corrects LR parsing** from folder names (e.g., `lr_0_01000` → 0.01)

**Input**: `Paper1_CV_Summary_Results.csv`, `fold_metrics` files  
**Output**: 
- `Combined_CV_Results_with_Folds.csv`
- `Combined_Test_Results_All_Models.csv`
- `Paper_Best_Results_Table.csv`

---

#### `07_Statistical_Analysis.py`
**Purpose**: Perform Friedman's ANOVA for statistical comparison of all models.

**Functionality**:
- Computes Friedman's test for each LR/BS combination
- Identifies best-performing method per configuration
- Counts wins by method category (Novel vs Baseline)
- Analyzes performance by learning rate, batch size, and metric

**Input**: `Combined_CV_Results_with_Folds.csv`  
**Output**: `Statistical_Analysis/Friedman_ANOVA_All_LRs.csv`

**Key Findings**:
- Novel VARDON models win in **~85%** of configurations
- Significant superiority over baselines demonstrated

---

#### `08_ROC_curves.py`
**Purpose**: Generate ROC curves from actual predictions.

**Functionality**:
- Loads per-fold predictions and true labels
- Computes FPR, TPR, and AUC for each method
- Plots ROC curves with method-specific colors and styles
- Saves figures in multiple formats

**Input**: `cv_runs/*/npy_files/fold*_predictions.npy`  
**Output**: 
- `Figure2_ROC_Curves_Revision.png`
- `Figure2_ROC_Curves_Revision.tiff`
- `Figure2_ROC_Curves_Revision.svg`

**Features**:
- 13 methods with distinct colors
- Differentiates: Baselines (dashed), VARDON (solid), Ablations (dotted)
- AUC values displayed in legend

---

### Figure Generation Scripts

#### `09_paper_Figures.py`
**Purpose**: Generate publication-ready figures for Oxford Bioinformatics (Revision).

**Figures Generated**:

| Figure | Description |
|--------|-------------|
| **Figure 1** | Hyperparameter Effects (LR × BS heatmap + bar plot) |
| **Figure 3** | Training and Validation Curves (2×2 subplots, 5 models) |
| **Figure 5** | Cross-Validation Boxplots (MCC distribution, 10-fold) |

**Functionality**:
- Loads combined CV and test results
- Generates heatmaps, bar plots, and boxplots
- Distinguishes VARDON variants with gold borders
- Saves in PNG, TIFF, and SVG formats

**Output**:
- `Figure1_Hyperparameter_Effects.png/tiff/svg`
- `Figure3_Training_Validation_Curves_5Models.png/tiff/svg`
- `Figure5_CV_Boxplots.png/tiff/svg`

---

#### `10_Figure_6_and_7.py`
**Purpose**: Extract and visualize learned dropout rates from VARDON models.

**Figure 6**: Distribution of Learned Dropout Rates
- Histogram showing distribution across all neurons
- Compares against fixed dropout rate (p=0.3)
- Per-method statistics

**Figure 7**: Dropout Rate vs Feature Importance
- **Focuses on VARDON_AdaptiveVD** (negative correlation)
- Scatter plot with trend line
- Pearson and Spearman correlation coefficients
- Shows learned dropout rates correlate negatively with feature importance

**Functionality**:
- Loads trained VARDON models
- Extracts dropout rates from RealVD and AdaptiveVD layers
- Extracts feature importance from first dense layer
- Computes correlation statistics
- Saves correlation summary table

**Input**: Trained VARDON model files  
**Output**:
- `Figure6_Dropout_Rates_Distribution.png/tiff`
- `Figure7_Dropout_vs_Importance.png/tiff`
- `Dropout_Rates_Summary.csv`
- `Dropout_Importance_Correlation.csv`

---

## Execution Order

### Data Processing (Once)

1. `01_Organize_Fish_Data.py` - Organize raw data
2. `02_uni_prot_data_processing.py` - Process embeddings with uncertainty filtering
3. `03_binary_data_set_with_species.py` - Create binary datasets
4. `04_Leave_one_species_out.py` - Create LOSO splits

### Model Training (Can be run on cluster)

5. `05_Model_Trainings1.py` - Main training script
   - Runs CV, LOSO, and test evaluation for all models
   - **Can be interrupted and resumed** using checkpoint system
   - Recommended to run overnight or on HPC

### Results Processing

6. `06_combine_files.py` - Combine CV results with fold data
7. `07_Statistical_Analysis.py` - Friedman's ANOVA
8. `08_ROC_curves.py` - Generate ROC curves

### Figure Generation

9. `09_paper_Figures.py` - Publication figures
10. `10_Figure_6_and_7.py` - Dropout rates analysis

---

## Key Output Tables

| File | Description |
|------|-------------|
| `Paper1_CV_Summary_Results.csv` | CV performance for all methods/configurations |
| `Paper1_Test_Results_All_Models.csv` | **All models** evaluated on test set |
| `LOSO_Results_All_Models.csv` | **All models** evaluated on LOSO |
| `Paper_Best_Results_Table.csv` | Best results per method |
| `Friedman_ANOVA_All_LRs.csv` | Statistical significance results |

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy tensorflow
