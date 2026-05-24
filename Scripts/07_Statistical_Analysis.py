# -*- coding: utf-8 -*-
"""
Statistical comparison of VARDON methods using ANOVA/Kruskal-Wallis tests
For Enzyme vs Non-enzyme Classification using 10-fold CV metrics
"""

import pandas as pd
import numpy as np
from scipy.stats import shapiro, levene, f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from itertools import combinations
import os

# =============================================
# CONFIGURATION
# =============================================

# Path to your combined fold metrics file
input_path = r"D:\zebfish\VARDON\Combined_Results\CV_all_models\ALL_METHODS_Fold_Metrics_Combined.csv"
output_dir = r"D:\zebfish\VARDON\Combined_Results\CV_all_models\Statistical_Analysis"

os.makedirs(output_dir, exist_ok=True)

# Metrics to compare (from your fold metrics)
metric_columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'AUC']

# Methods order for consistent display
methods_order = [
    'Logistic_Regression',
    'Gaussian_Dropout_NN',
    'Standard_Dropout',
    'VARDON_RealVD',
    'VARDON_AdaptiveVD',
    'VARDON_RealVD_Adaptive',
    'VARDON_Full',
    'VARDON_Light'
]

print("=" * 80)
print("STATISTICAL COMPARISON OF VARDON METHODS")
print("Using 10-Fold Cross-Validation Metrics")
print("=" * 80)

# =============================================
# 1. LOAD DATASET
# =============================================

print(f"\n📂 Loading data from: {input_path}")
df = pd.read_csv(input_path)

# Display column names to verify
print("\nAvailable columns in the dataset:")
print(df.columns.tolist())
print(f"\nDataset shape: {df.shape}")

# Display unique values
print(f"\nUnique Methods: {df['Method_Name'].unique()}")
print(f"Unique Learning Rates: {sorted(df['Learning_Rate'].unique())}")
print(f"Unique Batch Sizes: {sorted(df['Batch_Size'].unique())}")
print(f"Folds per configuration: {df.groupby(['Method_Name', 'Learning_Rate', 'Batch_Size']).size().iloc[0] if not df.empty else 'N/A'}")

# =============================================
# 2. EXTRACT LEARNING RATE AND BATCH SIZE
# =============================================

# The data already has Learning_Rate and Batch_Size columns
df['LR'] = df['Learning_Rate']
df['BS'] = df['Batch_Size']

print(f"\n✅ Learning Rate and Batch Size columns ready")

# =============================================
# 3. PARAMETER COMBINATIONS TO ANALYZE
# =============================================

param_combinations = df[['LR', 'BS']].drop_duplicates().dropna().sort_values(['LR', 'BS'])

print(f"\n📊 Parameter combinations to analyze: {len(param_combinations)}")
print(param_combinations.to_string(index=False))

# =============================================
# 4. STATISTICAL ANALYSIS FOR EACH CONFIGURATION
# =============================================

anova_results = []

for _, params in param_combinations.iterrows():
    lr = params['LR']
    bs = params['BS']

    # Filter subset
    subset = df[(df['LR'] == lr) & (df['BS'] == bs)]
    
    print(f"\n{'='*60}")
    print(f"Analyzing LR: {lr}, BS: {bs}")
    print(f"{'='*60}")
    print(f"Methods found: {subset['Method_Name'].unique().tolist()}")
    print(f"Folds per method: {subset.groupby('Method_Name')['Fold'].count().to_dict()}")

    for metric in metric_columns:
        data = subset[['Method_Name', metric]].dropna()
        
        # Check if we have enough data and groups
        if data['Method_Name'].nunique() < 2:
            print(f"\n  ⚠️ Skipping {metric}: Only {data['Method_Name'].nunique()} method(s) found")
            continue
            
        if len(data) < 6:
            print(f"\n  ⚠️ Skipping {metric}: Insufficient data ({len(data)} rows)")
            continue

        print(f"\n  📊 Analyzing {metric}: {len(data)} data points across {data['Method_Name'].nunique()} methods")

        # Test normality per group
        normal = True
        normality_results = []
        for method in data['Method_Name'].unique():
            vals = data[data['Method_Name'] == method][metric].dropna()
            if len(vals) < 3:
                normal = False
                normality_results.append(f"{method}: insufficient data ({len(vals)} points)")
                break
            shapiro_stat, shapiro_p = shapiro(vals)
            if shapiro_p < 0.05:
                normal = False
                normality_results.append(f"{method}: not normal (p={shapiro_p:.4f})")
            else:
                normality_results.append(f"{method}: normal (p={shapiro_p:.4f})")
        
        print(f"    Normality: {', '.join(normality_results)}")

        # Test equal variances
        groups = [group[metric].values for name, group in data.groupby('Method_Name')]
        if len(groups) < 2:
            continue
            
        levene_stat, levene_p = levene(*groups)
        equal_variances = levene_p > 0.05
        print(f"    Equal variances: {'Yes' if equal_variances else 'No'} (p={levene_p:.4f})")

        # Choose appropriate test
        if normal and equal_variances:
            # Parametric ANOVA
            try:
                model = ols(f'{metric} ~ C(Method_Name)', data=data).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                p_val = anova_table.iloc[0, 3]
                test_used = "ANOVA"
                print(f"    Using ANOVA, p-value: {p_val:.6f}")
            except Exception as e:
                print(f"    ❌ ANOVA failed: {e}")
                test_stat, p_val = kruskal(*groups)
                test_used = "Kruskal-Wallis (ANOVA fallback)"
                print(f"    Using Kruskal-Wallis instead, p-value: {p_val:.6f}")
        else:
            # Non-parametric Kruskal-Wallis test
            test_stat, p_val = kruskal(*groups)
            test_used = "Kruskal-Wallis"
            print(f"    Using Kruskal-Wallis, p-value: {p_val:.6f}")

        # Determine best method (highest mean)
        mean_scores = data.groupby('Method_Name')[metric].mean().sort_values(ascending=False)
        best_method = mean_scores.index[0]

        # Store result
        result = {
            'Learning_Rate': lr,
            'Batch_Size': bs,
            'Metric': metric,
            'Test_Used': test_used,
            'p_value': round(p_val, 6),
            'Best_Method': best_method,
            'Best_Mean': round(mean_scores.iloc[0], 4),
            'Significant': p_val < 0.05
        }

        # Add second best info if exists
        if len(mean_scores) > 1:
            result['Second_Best'] = mean_scores.index[1]
            result['Mean_Difference'] = round(mean_scores.iloc[0] - mean_scores.iloc[1], 4)
            result['Second_Best_Mean'] = round(mean_scores.iloc[1], 4)
        else:
            result['Second_Best'] = None
            result['Mean_Difference'] = None
            result['Second_Best_Mean'] = None

        result['Number_of_Methods'] = len(mean_scores)
        result['Total_Observations'] = len(data)
        
        anova_results.append(result)
        
        significance_text = "✅ SIGNIFICANT" if p_val < 0.05 else "❌ NOT significant"
        print(f"    {significance_text}: Best = {best_method} ({mean_scores.iloc[0]:.4f})")

# =============================================
# 5. OVERALL ANALYSIS (All configurations combined)
# =============================================

print(f"\n{'='*80}")
print("OVERALL ANALYSIS (All configurations combined)")
print('='*80)

overall_results = []

for metric in metric_columns:
    print(f"\n📊 Metric: {metric}")
    
    # Collect data for each method across all configurations
    method_data = {}
    for method in methods_order:
        method_df = df[df['Method_Name'] == method]
        if not method_df.empty:
            method_data[method] = method_df[metric].dropna().values
            print(f"  {method:<35}: n={len(method_data[method])}, mean={method_data[method].mean():.4f} ± {method_data[method].std():.4f}")
    
    if len(method_data) < 2:
        continue
    
    groups = list(method_data.values())
    
    # Kruskal-Wallis test (non-parametric, doesn't assume normality)
    h_stat, p_val = kruskal(*groups)
    
    # Find best method
    means = {method: values.mean() for method, values in method_data.items()}
    best_method = max(means, key=means.get)
    best_mean = means[best_method]
    
    overall_results.append({
        'Metric': metric,
        'p_value': round(p_val, 6),
        'Significant': p_val < 0.05,
        'Best_Method': best_method,
        'Best_Mean': round(best_mean, 4),
        'Number_of_Methods': len(method_data)
    })
    
    significance_text = "✅ SIGNIFICANT" if p_val < 0.05 else "❌ NOT significant"
    print(f"\n  Kruskal-Wallis: p={p_val:.6f} -> {significance_text}")
    print(f"  Best Method: {best_method} ({best_mean:.4f})")

# =============================================
# 6. SUMMARY STATISTICS
# =============================================

print(f"\n{'='*80}")
print("SUMMARY STATISTICS")
print('='*80)

if anova_results:
    anova_df = pd.DataFrame(anova_results)
    
    significant_comparisons = anova_df['Significant'].sum()
    total_comparisons = len(anova_df)
    print(f"\n📈 Total comparisons performed: {total_comparisons}")
    print(f"   Significant differences found: {significant_comparisons} ({significant_comparisons/total_comparisons*100:.1f}%)")
    
    # Show best methods by metric across all configurations
    print(f"\n📊 Best methods by metric (across all configurations):")
    for metric in metric_columns:
        metric_data = anova_df[anova_df['Metric'] == metric]
        if not metric_data.empty:
            # Get most frequent best method for this metric
            best_method_counts = metric_data['Best_Method'].value_counts()
            top_method = best_method_counts.index[0]
            top_count = best_method_counts.iloc[0]
            print(f"  {metric}: {top_method} (best in {top_count}/{len(metric_data)} configurations)")
    
    # Show best overall metric values
    print(f"\n📊 Best achieved values across all configurations:")
    for metric in metric_columns:
        max_value = anova_df[anova_df['Metric'] == metric]['Best_Mean'].max()
        best_config = anova_df[anova_df['Metric'] == metric].loc[anova_df[anova_df['Metric'] == metric]['Best_Mean'].idxmax()]
        print(f"  {metric}: {max_value:.4f} ({best_config['Best_Method']}, LR={best_config['Learning_Rate']}, BS={best_config['Batch_Size']})")

# =============================================
# 7. SAVE RESULTS
# =============================================

print(f"\n{'='*80}")
print("SAVING RESULTS")
print('='*80)

# Save configuration-wise results
if anova_results:
    anova_df = pd.DataFrame(anova_results)
    anova_path = os.path.join(output_dir, "Statistical_Comparison_By_Configuration.csv")
    anova_df.to_csv(anova_path, index=False)
    print(f"✅ Configuration-wise results saved to: {anova_path}")

# Save overall results
if overall_results:
    overall_df = pd.DataFrame(overall_results)
    overall_path = os.path.join(output_dir, "Overall_Statistical_Comparison.csv")
    overall_df.to_csv(overall_path, index=False)
    print(f"✅ Overall results saved to: {overall_path}")

# Create a summary table by metric
if anova_results:
    summary_df = anova_df.groupby(['Metric', 'Best_Method']).agg({
        'Best_Mean': 'mean',
        'Significant': 'sum',
        'p_value': 'count'
    }).rename(columns={'p_value': 'Count'}).reset_index()
    
    summary_df = summary_df.sort_values(['Metric', 'Best_Mean'], ascending=[True, False])
    summary_path = os.path.join(output_dir, "Statistical_Comparison_Summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Summary table saved to: {summary_path}")
    
    print("\n📊 SUMMARY TABLE (Best Method per Metric):")
    print(summary_df.to_string(index=False))

# =============================================
# 8. PRINT FINAL SUMMARY
# =============================================

print(f"\n{'='*80}")
print("FINAL SUMMARY FOR PAPER")
print('='*80)

print("\n📊 Cross-Validation Performance (Mean ± Std across all configurations):")
method_summary = df.groupby('Method_Name').agg({
    'Accuracy': ['mean', 'std'],
    'Precision': ['mean', 'std'],
    'Recall': ['mean', 'std'],
    'F1': ['mean', 'std'],
    'MCC': ['mean', 'std'],
    'AUC': ['mean', 'std']
}).round(4)

# Reorder by MCC
method_summary = method_summary.reindex(methods_order)
print(method_summary)

print(f"\n{'='*80}")
print("✅ STATISTICAL ANALYSIS COMPLETE!")
print(f"📁 All results saved to: {output_dir}")
print("=" * 80)