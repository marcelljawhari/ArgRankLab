# generate_report.py

import os
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd

# --- Configuration ---
RESULTS_DIR = os.path.join("data", "results")
PROPERTIES_FILE = os.path.join("data", "framework_properties.csv")
OUTPUT_FILE = os.path.join("data", "report.csv")

FAST_SEMANTICS = ["Cat", "Dbs", "p-Stable", "p-Admissible"]
SLOW_SEMANTICS = ["Ser", "p-Complete", "p-Grounded", "p-Ideal", "p-Preferred"]
ALL_SEMANTICS = sorted(FAST_SEMANTICS + SLOW_SEMANTICS)

# Define the analysis groups for the final hybrid report
HYBRID_ANALYSIS_GROUPS = [
    {'title': 'Overall Hybrid Analysis', 'filters': []},
    {'title': 'Cyclic Hybrid Analysis', 'filters': [('cyclicity', '==', 'Cyclic')]},
    {'title': 'Acyclic Hybrid Analysis', 'filters': [('cyclicity', '==', 'Acyclic')]},
    {'title': 'Sparse Hybrid Analysis', 'filters': [('density_group', '==', 'Sparse')]},
    {'title': 'Dense Hybrid Analysis', 'filters': [('density_group', '==', 'Dense')]},
]

# --- Helper Functions ---

def format_report_for_group(kendall_data: dict, spearman_data: dict, title: str, semantics_list: list, count_all: int, count_tweety: int) -> str:
    """Takes aggregated data and returns a formatted string block for the report."""
    if count_all == 0 and count_tweety == 0:
        return f"--- No data available for group: {title} ---\n"

    report_parts = []
    report_title = (
        f"{title} (Fast-vs-Fast on {count_all} AFs, others on {count_tweety} Tweety AFs)"
    )

    def build_matrix_string(data: dict, metric_title: str, agg_func) -> str:
        """Builds a CSV matrix string for a given aggregation function."""
        matrix = pd.DataFrame(index=semantics_list, columns=semantics_list, dtype=float)
        for n1, n2 in itertools.product(semantics_list, semantics_list):
            if n1 == n2:
                if agg_func in [np.mean, np.median]:
                    matrix.loc[n1, n2] = 1.0
                else: # np.std
                    matrix.loc[n1, n2] = 0.0
            else:
                corrs = data.get((n1, n2), []) or data.get((n2, n1), [])
                if corrs:
                    matrix.loc[n1, n2] = agg_func(corrs)
        return f"--- {metric_title} ---\n{matrix.to_csv()}\n"

    report_parts.append(build_matrix_string(kendall_data, f"{report_title} - Kendall's Tau (Average)", np.mean))
    report_parts.append(build_matrix_string(kendall_data, f"{report_title} - Kendall's Tau (Median)", np.median))
    report_parts.append(build_matrix_string(kendall_data, f"{report_title} - Kendall's Tau (Standard Deviation)", np.std))
    report_parts.append(build_matrix_string(spearman_data, f"{report_title} - Spearman's Rho (Average)", np.mean))
    report_parts.append(build_matrix_string(spearman_data, f"{report_title} - Spearman's Rho (Median)", np.median))
    report_parts.append(build_matrix_string(spearman_data, f"{report_title} - Spearman's Rho (Standard Deviation)", np.std))
    
    return "\n".join(report_parts)

def aggregate_correlations(frameworks_df: pd.DataFrame) -> (dict, dict, int):
    """Reads result files for a given set of frameworks and aggregates the correlations."""
    kendall_agg = defaultdict(list)
    spearman_agg = defaultdict(list)
    processed_count = 0
    for _, row in frameworks_df.iterrows():
        framework_name = row['framework_name']
        kendall_path = os.path.join(RESULTS_DIR, framework_name.replace('.af', '_kendall.csv'))
        spearman_path = os.path.join(RESULTS_DIR, framework_name.replace('.af', '_spearman.csv'))
        
        if os.path.exists(kendall_path) and os.path.exists(spearman_path):
            processed_count += 1
            try:
                df_k = pd.read_csv(kendall_path, index_col=0)
                df_s = pd.read_csv(spearman_path, index_col=0)
                for (c1, c2) in itertools.combinations(df_k.columns, 2):
                    if c1 in df_k.index and c2 in df_k.index:
                        kendall_agg[(c1, c2)].append(df_k.loc[c1, c2])
                        spearman_agg[(c1, c2)].append(df_s.loc[c1, c2])
            except Exception as e:
                print(f"Warning: Could not read result for {framework_name}. Reason: {e}")
                processed_count -= 1
    return kendall_agg, spearman_agg, processed_count

# --- Main Analysis Logic ---

def main():
    """Main function to generate a single, hybrid report.csv file."""
    print("=" * 60)
    print("  Generating Final Hybrid Analysis Report  ")
    print("=" * 60)

    try:
        properties_df = pd.read_csv(PROPERTIES_FILE)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at '{PROPERTIES_FILE}'")
        print("Please run 'classify_frameworks.py' first.")
        return

    print(f"Loaded metadata for {len(properties_df)} frameworks.")
    report_content = []

    for group_config in HYBRID_ANALYSIS_GROUPS:
        title = group_config['title']
        filters = group_config['filters']
        
        print(f"\nProcessing group: {title}...")
        
        subset_df = properties_df.copy()
        for col, op, val in filters:
            if op == '==': subset_df = subset_df[subset_df[col] == val]
            elif op == '!=': subset_df = subset_df[subset_df[col] != val]
            else: raise ValueError(f"Unsupported operator '{op}'")

        all_filtered_frameworks = subset_df
        tweety_filtered_frameworks = subset_df[subset_df['source_dataset'] == 'tweety']

        k_all, s_all, count_all = aggregate_correlations(all_filtered_frameworks)
        k_tweety, s_tweety, count_tweety = aggregate_correlations(tweety_filtered_frameworks)

        k_hybrid = defaultdict(list)
        s_hybrid = defaultdict(list)
        for sem1, sem2 in itertools.combinations(ALL_SEMANTICS, 2):
            if sem1 in FAST_SEMANTICS and sem2 in FAST_SEMANTICS:
                k_hybrid[(sem1, sem2)] = k_all.get((sem1, sem2), [])
                s_hybrid[(sem1, sem2)] = s_all.get((sem1, sem2), [])
            else:
                k_hybrid[(sem1, sem2)] = k_tweety.get((sem1, sem2), [])
                s_hybrid[(sem1, sem2)] = s_tweety.get((sem1, sem2), [])
        
        report_block = format_report_for_group(k_hybrid, s_hybrid, title, ALL_SEMANTICS, count_all, count_tweety)
        report_content.append(report_block)

    with open(OUTPUT_FILE, 'w') as f:
        f.write("\n\n".join(report_content))

    print(f"\nAnalysis complete. Full hybrid report saved to '{OUTPUT_FILE}'")
    print("=" * 60)

if __name__ == '__main__':
    main()