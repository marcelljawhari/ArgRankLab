# main.py

import os
import time
import multiprocessing as mp
import networkx as nx

from util.af_parser import parse_af_file
from semantics.ser import Ser

# --- Configuration ---
TIMEOUT_SECONDS = 600  # 10 minutes

def ser_worker(af: nx.DiGraph, queue: mp.Queue):
    """
    A dedicated worker function to run the Ser calculation.
    This runs in a separate process to allow for a timeout.
    """
    try:
        calculator = Ser(af=af)
        ranking_groups = calculator.get_ranking()
        queue.put(ranking_groups)
    except Exception as e:
        # Put the exception in the queue so the main process can report it.
        queue.put(e)

def convert_groups_to_ranking(ranking_groups: list, all_args: list) -> list:
    """
    Converts a list of ranked groups (sets of equally-ranked args) into a single,
    sorted list of arguments. Arguments in earlier groups come first.
    """
    ranked_args = [arg for group in ranking_groups for arg in sorted(list(group))]
    present_args_set = set(ranked_args)
    missing_args = [arg for arg in all_args if arg not in present_args_set]
    return ranked_args + missing_args

def run_ser_on_framework(framework_path: str):
    """
    Runs the Ser-based semantics on a single argumentation framework with a timeout.
    """
    if not os.path.exists(framework_path):
        print(f"Error: Framework file not found at {framework_path}")
        return

    print("="*60)
    print(f"Processing framework: {os.path.basename(framework_path)}")
    print("="*60)
    
    try:
        arg_framework = parse_af_file(framework_path)
        all_args_sorted = sorted(list(arg_framework.nodes()))
        
        print(f"Graph loaded with {arg_framework.number_of_nodes()} nodes and {arg_framework.number_of_edges()} edges.")
        print(f"Running Ser semantics with a {TIMEOUT_SECONDS}s timeout...", end="", flush=True)

        # --- Multiprocessing for Timeout ---
        result_queue = mp.Queue()
        process = mp.Process(target=ser_worker, args=(arg_framework, result_queue))
        
        start_time = time.time()
        process.start()
        process.join(TIMEOUT_SECONDS)
        end_time = time.time()

        if process.is_alive():
            # Process is still running, so the timeout was reached.
            process.terminate()
            process.join()
            print(" TIMEOUT!")
            print(f"Calculation exceeded the {TIMEOUT_SECONDS} second limit.")
        else:
            # Process finished in time.
            result = result_queue.get()
            if isinstance(result, Exception):
                # An error occurred inside the worker process.
                print(" ERROR!")
                raise result

            print(f" done in {end_time - start_time:.4f} seconds.")
            ranking_groups = result
            final_ranking = convert_groups_to_ranking(ranking_groups, all_args_sorted)
            
            # --- Display the results ---
            print("\n--- Final Argument Ranking ---")
            if len(final_ranking) > 15:
                print(f"Top 15 arguments: {final_ranking[:15]}")
                print(f"(... and {len(final_ranking) - 15} more arguments)")
            else:
                print(final_ranking)

        print("\n" + "="*60 + "\n")

    except Exception as e:
        print(f"\n!!! An error occurred while processing {framework_path}: {e}\n")


if __name__ == '__main__':
    # It's good practice to use 'fork' for multiprocessing on Unix-like systems.
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        print("Using default multiprocessing start method.")

    # Directory where the generated benchmarks are stored.
    benchmark_dir = os.path.join("data", "benchmarks_generated")
    
    print("===================================================")
    print("  Running Ser Semantics on all Generated Benchmarks  ")
    print("===================================================\n")

    if not os.path.exists(benchmark_dir):
        print(f"Error: Benchmark directory not found at '{benchmark_dir}'")
        print("Please run the 'generate_benchmarks.py' script first.")
    else:
        framework_files = sorted([f for f in os.listdir(benchmark_dir) if f.endswith(".af")])

        if not framework_files:
            print(f"No '.af' benchmark files found in '{benchmark_dir}'.")
        else:
            print(f"Found {len(framework_files)} benchmark(s) to process.\n")
            for filename in framework_files:
                full_path = os.path.join(benchmark_dir, filename)
                run_ser_on_framework(full_path)
            
            print("All benchmarks have been processed.")