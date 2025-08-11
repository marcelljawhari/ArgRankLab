# run_semantics.py

import os
import time
import multiprocessing as mp
import itertools
from collections import defaultdict

import networkx as nx
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from util.af_parser import parse_af_file
from semantics.cat import Cat
from semantics.dbs import Dbs
from semantics.ser import Ser
from semantics.prob.prob_admissible import ProbAdmissible
from semantics.prob.prob_complete import ProbComplete
from semantics.prob.prob_grounded import ProbGrounded
from semantics.prob.prob_ideal import ProbIdeal
from semantics.prob.prob_preferred import ProbPreferred
from semantics.prob.prob_stable import ProbStable


# --- Configuration ---
BENCHMARK_DIRS = [
    os.path.join("data", "benchmarks_tweety"),
    os.path.join("data", "benchmarks2023", "main")
]
RESULTS_DIR = os.path.join("data", "results")
TIMEOUT_SECONDS = 600

FAST_SEMANTICS = {
    "Cat": Cat,
    "Dbs": Dbs,
    "p-Stable": ProbStable, # This is the analytical one
    "p-Admissible": ProbAdmissible, # This is the analytical one
}
SLOW_SEMANTICS = {
    "Ser": Ser,
    "p-Complete": ProbComplete,
    "p-Ideal": ProbIdeal,
    "p-Grounded": ProbGrounded,
    "p-Preferred": ProbPreferred
}

# --- Worker for Multiprocessing Timeout ---

def semantics_worker(name: str, sem_class, af: nx.DiGraph, queue: mp.Queue):
    """A generic worker to run any semantics calculation in a separate process."""
    try:
        calculator = sem_class(af)
        if hasattr(calculator, 'get_scores'):
            result = calculator.get_scores()
        elif hasattr(calculator, 'get_ranking'):
            result = calculator.get_ranking()
        else:
            raise NotImplementedError(f"Semantics class {name} has no get_scores or get_ranking method.")
        queue.put(result)
    except Exception as e:
        queue.put(e)

# --- Ranking Normalization ---

def normalize_ranking(result, all_args_sorted: list) -> list:
    """Converts different semantics outputs into a single, ordered list of arguments."""
    if isinstance(result, dict):
        return sorted(result, key=lambda arg: (-result.get(arg, -float('inf')), int(arg)))
    if isinstance(result, list):
        ranked_args = [arg for group in result for arg in sorted(list(group), key=int)]
        present_args_set = set(ranked_args)
        missing_args = [arg for arg in all_args_sorted if arg not in present_args_set]
        return ranked_args + sorted(missing_args, key=int)
    raise TypeError(f"Unsupported result type for normalization: {type(result)}")

# --- File Discovery ---

def find_framework_files(root_dirs: list) -> list:
    """Finds all .af files in a list of directories and their subdirectories."""
    framework_paths = []
    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            print(f"Warning: Benchmark directory not found at '{root_dir}'")
            continue
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(".af"):
                    framework_paths.append(os.path.join(dirpath, filename))
    return sorted(framework_paths)


# --- HELPER FUNCTION ---

def create_and_save_matrix(rankings: dict, result_path: str, metric_func, metric_name: str):
    """Calculates and saves a correlation matrix for a given metric."""
    semantics_names = sorted(list(rankings.keys()))
    matrix = pd.DataFrame(index=semantics_names, columns=semantics_names, dtype=float)

    for (name1, rank1), (name2, rank2) in itertools.combinations(rankings.items(), 2):
        corr, _ = metric_func(rank1, rank2)
        matrix.loc[name1, name2] = corr
        matrix.loc[name2, name1] = corr
    
    for name in semantics_names:
        matrix.loc[name, name] = 1.0

    print(f"  Saving {metric_name} results to {result_path}")
    matrix.to_csv(result_path)


# --- Main Execution Logic ---

def main():
    """Main function to run the full correlation analysis."""
    print("=" * 60)
    print("  Argumentation Semantics Correlation Analysis  ")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    framework_paths = find_framework_files(BENCHMARK_DIRS)
    if not framework_paths:
        print(f"\nError: No '.af' files found in any benchmark directories.")
        return

    print(f"\nFound {len(framework_paths)} total benchmark frameworks to process.")
    
    processed_count = 0
    timeout_count = 0
    already_done_count = 0

    for i, framework_path in enumerate(framework_paths):
        framework_name = os.path.basename(framework_path)
        base_result_name = framework_name.replace('.af', '')
        kendall_path = os.path.join(RESULTS_DIR, f"{base_result_name}_kendall.csv")
        spearman_path = os.path.join(RESULTS_DIR, f"{base_result_name}_spearman.csv")
        timeout_marker_path = os.path.join(RESULTS_DIR, f"{base_result_name}.timeout")

        print(f"\n--- [{i+1}/{len(framework_paths)}] Checking: {framework_name} ---")

        if os.path.exists(timeout_marker_path):
            print("  Previously timed out. Skipping.")
            timeout_count += 1
            continue
        if os.path.exists(kendall_path) and os.path.exists(spearman_path):
            print("  Result already exists. Skipping.")
            already_done_count += 1
            continue
        
        if "benchmarks_tweety" in framework_path:
            semantics_to_run_now = {**FAST_SEMANTICS, **SLOW_SEMANTICS}
        else:
            semantics_to_run_now = FAST_SEMANTICS
        
        try:
            arg_framework = parse_af_file(framework_path)
            all_args_sorted = sorted(list(arg_framework.nodes()), key=int)
        except Exception as e:
            print(f"  ERROR: Could not parse file. Skipping. Reason: {e}")
            continue

        rankings = {}
        framework_timed_out = False
        for name, sem_class in semantics_to_run_now.items():
            print(f"  - Running {name}...")
            result_queue = mp.Queue()
            process = mp.Process(target=semantics_worker, args=(name, sem_class, arg_framework, result_queue))
            start_time = time.time()
            process.start()
            process.join(TIMEOUT_SECONDS)
            end_time = time.time()

            if process.is_alive():
                process.terminate(); process.join()
                print("    TIMEOUT!")
                with open(timeout_marker_path, 'w') as f:
                    f.write(f"Timeout occurred on {time.ctime()} with semantics: {name}")
                framework_timed_out = True
                timeout_count += 1
                break

            result = result_queue.get()
            if isinstance(result, Exception):
                print(f"    ERROR! ({result})")
                rankings[name] = None
            else:
                print(f"    done ({end_time - start_time:.2f}s)")
                rankings[name] = normalize_ranking(result, all_args_sorted)
        
        if framework_timed_out:
            continue

        valid_rankings = {name: rank for name, rank in rankings.items() if rank is not None}
        if len(valid_rankings) < 2:
            print("  Skipping correlation (not enough successful runs).")
            continue
        
        create_and_save_matrix(valid_rankings, kendall_path, kendalltau, "Kendall's Tau")
        create_and_save_matrix(valid_rankings, spearman_path, spearmanr, "Spearman's Rho")
        processed_count += 1

    # --- Part 2: Aggregate all results ---
    print("\n" + "=" * 60)
    print("  All frameworks checked. Generating final summary.  ")
    print("=" * 60)
    
    print("Run Summary:")
    print(f"  - Frameworks Processed in this run: {processed_count}")
    print(f"  - Frameworks Skipped (already done):  {already_done_count}")
    print(f"  - Frameworks Skipped (timeout):       {timeout_count}")
    
    result_files = os.listdir(RESULTS_DIR)
    kendall_files = [os.path.join(RESULTS_DIR, f) for f in result_files if f.endswith('_kendall.csv')]
    spearman_files = [os.path.join(RESULTS_DIR, f) for f in result_files if f.endswith('_spearman.csv')]

    def build_summary_matrix(files: list, all_sem_names: list) -> pd.DataFrame:
        if not files: return pd.DataFrame(index=all_sem_names, columns=all_sem_names)
        agg_data = defaultdict(list)
        for res_path in files:
            df = pd.read_csv(res_path, index_col=0)
            for (col1, col2) in itertools.combinations(df.columns, 2):
                 if col1 in df.index and col2 in df.index:
                    corr_val = df.loc[col1, col2]
                    if pd.notna(corr_val):
                        agg_data[(col1, col2)].append(corr_val)
                        agg_data[(col2, col1)].append(corr_val)
        summary_matrix = pd.DataFrame(index=all_sem_names, columns=all_sem_names, dtype=float)
        for name1 in all_sem_names:
            for name2 in all_sem_names:
                if name1 == name2:
                    summary_matrix.loc[name1, name2] = 1.0
                else:
                    corrs = agg_data.get((name1, name2), [])
                    if corrs:
                        summary_matrix.loc[name1, name2] = sum(corrs) / len(corrs)
                    else:
                        summary_matrix.loc[name1, name2] = None
        return summary_matrix

    all_semantics_names = sorted(list({**FAST_SEMANTICS, **SLOW_SEMANTICS}.keys()))
    pd.set_option('display.float_format', '{:.4f}'.format)

    print("\nAverage Kendall's Tau Correlation Matrix (All Valid Frameworks):\n")
    kendall_summary = build_summary_matrix(kendall_files, all_semantics_names)
    print(kendall_summary.to_string())
    
    print("\n\nAverage Spearman's Rho Correlation Matrix (All Valid Frameworks):\n")
    spearman_summary = build_summary_matrix(spearman_files, all_semantics_names)
    print(spearman_summary.to_string())

    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    try:
        if mp.get_start_method() != "fork":
            mp.set_start_method("fork", force=True)
    except (RuntimeError, AttributeError):
        print("Warning: Could not set multiprocessing start method to 'fork'. Using default.")
    
    main()