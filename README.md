# VARDON: Enzyme Classification Pipeline

This repository contains Python scripts for VARDON (Variational Adaptive Real Dropout Neural Network), a pipeline designed for enzyme classification across 12 fish species.

---

## File Descriptions

### 01_organize_fish_data.py
**Purpose:** Organize raw UniProt downloads into structured format.

**Functionality:**
- Scans raw data folder with species subfolders.
- Identifies HDF5 embedding files and TSV annotation files.
- Copies HDF5 files as `embeddings.h5` and TSV files as `{species}_annotations.tsv`.
- Creates `metadata.csv` and `master_summary.csv`.

**Input:** `D:\zebfish\`  
**Output:** `D:\zebfish_organized\`

---

### 02_uniprot_data_processor.py
**Purpose:** Process embeddings and annotations to create labeled datasets.

**Functionality:**
- Loads HDF5 embeddings and TSV annotations.
- Labels proteins as `Enzyme` or `Non-enzyme`.
- Merges embeddings with labels by UniProt ID.
- Saves clean dataset for deep learning.

**Input:** `D:\zebfish_organized\`  
**Output:** `D:\zebfish_processed_results\combined_data\clean_fish_dataset_for_dl.csv`

---

### 03_prepare_binary_dataset.py
**Purpose:** Convert labeled dataset into binary format.

**Functionality:**
- Removes metadata columns.
- Converts labels to numeric: `Enzyme` → 1, `Non-enzyme` → 0.
- Renames classification column to `Label`.

**Input:** `clean_fish_dataset_for_dl.csv`  
**Output:** `binary_classification_dataset.csv`

---

### 04_vardon_experiments.py
**Purpose:** Run VARDON variants and baseline models.

**Functionality:**
- Implements 5 VARDON variants and 3 baseline models.
- Performs 90/10 train-test split.
- Runs 10-fold cross-validation.
- Evaluates best model on test set.
- Saves training histories, metrics, and trained models.

**Input:** `binary_classification_dataset.csv`  
**Output:** Result directories for each learning rate (0.01, 0.001, 0.0001)

---

### 05_combine_fold_metrics.py
**Purpose:** Combine metrics from cross-validation folds.

**Functionality:**
- Scans result directories for `Fold_Metrics.csv`.
- Calculates mean and standard deviation across 10 folds.
- Saves combined results for analysis.

**Input:** VARDON result folders  
**Output:** `ALL_METHODS_Fold_Metrics_Combined.csv`

---

### 06_combine_results.py
**Purpose:** Combine results across multiple learning rates.

**Functionality:**
- Combines CV summary and test results.
- Produces `Paper_Best_Results_Table.csv`.
- Identifies overall best-performing model.

**Input:** All LR result directories  
**Output:** Combined result tables

---

### 07_statistical_analysis.py
**Purpose:** Statistical analysis of models.

**Functionality:**
- Tests normality and homogeneity of variances.
- Performs Kruskal-Wallis and pairwise tests.
- Saves statistical comparison tables.

**Input:** `ALL_METHODS_Fold_Metrics_Combined.csv`  
**Output:** Statistical analysis CSVs

---

### 08_generate_roc_curve.py
**Purpose:** Generate ROC curves.

**Functionality:**
- Computes FPR, TPR, and AUC for each method.
- Plots ROC curves for all models.
- Saves figures in PNG and TIFF.

**Input:** Predictions and true labels (`.npy`)  
**Output:** ROC curve figures

---

### 09_generate_figures.py
**Purpose:** Create publication-ready figures.

**Figures Generated:**
1. Hyperparameter effects
2. Training and validation curves
3. VARDON variant performance
4. Cross-validation boxplots

**Input:** Combined fold metrics + training histories  
**Output:** Figures in PNG and TIFF

---

## Execution Order

1. `01_organize_fish_data.py`
2. `02_uniprot_data_processor.py`
3. `03_prepare_binary_dataset.py`
4. `04_vardon_experiments.py`
5. `05_combine_fold_metrics.py`
6. `06_combine_results.py`
7. `07_statistical_analysis.py`
8. `08_generate_roc_curve.py`
9. `09_generate_figures.py`

---

## Directory Structure
