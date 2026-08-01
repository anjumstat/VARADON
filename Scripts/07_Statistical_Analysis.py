# -*- coding: utf-8 -*-
"""
Friedman's ANOVA for ALL LRs (with complete fold data)
"""

import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
import os
import ast
import warnings
warnings.filterwarnings("ignore")

# Load data with folds
input_path = r"D:\zebfish\revision\Combined_Results\Combined_CV_Results_with_Folds.csv"
output_dir = r"D:\zebfish\revision\Combined_Results\Statistical_Analysis"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_path)
print(f"✅ Loaded {len(df)} rows")

# Methods
novel_models = [
    'VARDON_RealVD', 'VARDON_AdaptiveVD', 'VARDON_RealVD_Adaptive',
    'VARDON_Full', 'VARDON_Light',
    'VARDON_Full_No_Gate', 'VARDON_Full_No_Sparsity', 'VARDON_Full_No_Both'
]

all_results = []

for lr in [0.0001, 0.001, 0.01]:
    for bs in [32, 64, 128]:
        subset = df[(df['LR'] == lr) & (df['BS'] == bs)]
        
        if subset.empty:
            continue
        
        for metric in ['fold_mcc_values', 'fold_accuracy_values', 'fold_f1_values', 'fold_auc_values']:
            metric_name = metric.replace('fold_', '').replace('_values', '')
            
            # Get data for each method
            method_data = {}
            for method in subset['Method'].unique():
                row = subset[subset['Method'] == method]
                if not row.empty:
                    vals = row[metric].values[0]
                    if isinstance(vals, str):
                        vals = ast.literal_eval(vals)
                    if isinstance(vals, list) and len(vals) == 10:
                        method_data[method] = np.array(vals)
            
            if len(method_data) < 3:
                continue
            
            methods = list(method_data.keys())
            data_matrix = np.array([method_data[m] for m in methods])
            
            # Friedman's test
            try:
                stat, p_val = friedmanchisquare(*data_matrix)
                significant = p_val < 0.05
                
                # Find best method
                means = {m: np.mean(method_data[m]) for m in methods}
                best_method = max(means, key=means.get)
                best_mean = means[best_method]
                
                all_results.append({
                    'LR': lr,
                    'BS': bs,
                    'Metric': metric_name,
                    'p_value': p_val,
                    'Significant': significant,
                    'Best_Method': best_method,
                    'Best_Mean': best_mean,
                    'Is_Novel': best_method in novel_models,
                    'Num_Methods': len(methods)
                })
                
                print(f"LR={lr}, BS={bs}, {metric_name}: p={p_val:.6f} -> {'✅' if significant else '❌'} Significant")
                print(f"  Best: {best_method} ({best_mean:.4f})")
                
            except Exception as e:
                print(f"Error for LR={lr}, BS={bs}, {metric_name}: {e}")

# Save results
if all_results:
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(output_dir, "Friedman_ANOVA_All_LRs.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to: {results_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Novel wins
    novel_wins = results_df['Is_Novel'].sum()
    total = len(results_df)
    print(f"\n🏆 Novel Models Win: {novel_wins}/{total} ({novel_wins/total*100:.1f}%)")
    print(f"   Baseline Models Win: {total - novel_wins}/{total} ({(total - novel_wins)/total*100:.1f}%)")
    
    # By LR
    print("\n📊 Wins by Learning Rate:")
    for lr in [0.0001, 0.001, 0.01]:
        lr_data = results_df[results_df['LR'] == lr]
        if not lr_data.empty:
            novel = lr_data['Is_Novel'].sum()
            total_lr = len(lr_data)
            print(f"  LR={lr}: Novel={novel}/{total_lr} ({novel/total_lr*100:.1f}%)")
    
    # By BS
    print("\n📊 Wins by Batch Size:")
    for bs in [32, 64, 128]:
        bs_data = results_df[results_df['BS'] == bs]
        if not bs_data.empty:
            novel = bs_data['Is_Novel'].sum()
            total_bs = len(bs_data)
            print(f"  BS={bs}: Novel={novel}/{total_bs} ({novel/total_bs*100:.1f}%)")
    
    # By Metric
    print("\n📊 Wins by Metric:")
    for metric in ['mcc', 'accuracy', 'f1', 'auc']:
        metric_data = results_df[results_df['Metric'] == metric]
        if not metric_data.empty:
            novel = metric_data['Is_Novel'].sum()
            total_metric = len(metric_data)
            print(f"  {metric}: Novel={novel}/{total_metric} ({novel/total_metric*100:.1f}%)")
    
    # Best method
    print("\n🏆 Best Method by Wins:")
    win_counts = results_df['Best_Method'].value_counts()
    for method, count in win_counts.items():
        is_novel = "✅ Novel" if method in novel_models else "❌ Baseline"
        print(f"  {method}: {count} wins ({is_novel})")