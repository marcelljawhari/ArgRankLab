import numpy as np
from scipy.sparse import lil_matrix, identity
from typing import List, Dict, Tuple
import os
import networkx as nx
import multiprocessing
import queue
import time

def parse_af_file(file_path: str) -> nx.DiGraph:
    """
    Parses an argumentation framework from a file in the ICCMA '.af' format.
    This function is based on the provided util/af_parser.py.

    Args:
        file_path: The path to the .af file.

    Returns:
        A networkx.DiGraph object representing the argumentation framework.
    """
    graph = nx.DiGraph()
    num_args_declared = 0

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith('#'):
                    continue

                if line.startswith('p af'):
                    try:
                        num_args_declared = int(line.split()[2])
                        graph.add_nodes_from([str(i) for i in range(1, num_args_declared + 1)])
                    except (IndexError, ValueError):
                        print(f"Warning: Could not parse p-line: {line}")
                else:
                    try:
                        attacker, attacked = line.split()
                        graph.add_edge(attacker, attacked)
                    except ValueError:
                        print(f"Warning: Skipping malformed attack line: {line}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return graph

    if num_args_declared > 0 and len(graph.nodes()) > num_args_declared:
         print(f"Warning: More arguments ({len(graph.nodes())}) found than declared ({num_args_declared}).")

    return graph

def calculate_dbs_worker(result_queue: multiprocessing.Queue, arguments: List[str], attacks: List[Tuple[str, str]], max_len: int):
    """
    A wrapper for calculate_dbs to be run in a separate process.
    Puts the result or an exception into the queue.
    """
    try:
        # This is the core computational function
        ranked_args, _ = calculate_dbs(arguments, attacks, max_len)
        result_queue.put(ranked_args)
    except Exception as e:
        result_queue.put(e)


def calculate_dbs(arguments: List[str], attacks: List[Tuple[str, str]], max_len: int = 0) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Calculates the ranking of arguments based on the Discussion-Based Semantics,
    using the path-based definition from the provided thesis.
    """
    num_args = len(arguments)
    if num_args == 0:
        return [], {}

    if max_len <= 0:
        max_len = num_args

    arg_to_idx = {arg: i for i, arg in enumerate(arguments)}
    adj_matrix = lil_matrix((num_args, num_args), dtype=np.int64)
    for attacker, attacked in attacks:
        if attacker in arg_to_idx and attacked in arg_to_idx:
            adj_matrix[arg_to_idx[attacker], arg_to_idx[attacked]] = 1

    adj_matrix_T = adj_matrix.transpose().tocsr()
    discussion_vectors: Dict[str, List[int]] = {arg: [] for arg in arguments}
    current_power = adj_matrix_T.copy()

    for path_length in range(1, max_len + 1):
        num_paths_per_arg = current_power.sum(axis=1).A1

        if path_length % 2 != 0:
            dis_values = num_paths_per_arg
        else:
            dis_values = -num_paths_per_arg

        for i in range(num_args):
            arg = arguments[i]
            discussion_vectors[arg].append(int(dis_values[i]))

        if current_power.nnz == 0:
            break
        
        current_power = current_power.dot(adj_matrix_T)

    sorted_arguments = sorted(arguments, key=lambda arg: discussion_vectors[arg])
    return sorted_arguments, discussion_vectors

def run_full_analysis():
    """
    Main function to run the Dbs analysis on the entire dataset.
    """
    benchmark_dirs = ['data/benchmarks_tweety', 'data/benchmarks2023/main']
    timeout_seconds = 600  # 10 minutes

    # --- Statistics Counters ---
    success_count = 0
    timeout_count = 0
    error_count = 0
    skipped_count = 0
    total_files = 0
    timed_out_files = []

    # --- 1. Find all .af files ---
    af_files = []
    for directory in benchmark_dirs:
        if not os.path.isdir(directory):
            print(f"Warning: Benchmark directory not found: {directory}")
            continue
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.af'):
                    af_files.append(os.path.join(root, file))
    
    total_files = len(af_files)
    print(f"Found {total_files} .af files to process.\n")

    # --- 2. Process each file ---
    for i, file_path in enumerate(af_files):
        print(f"[{i+1}/{total_files}] Processing: {os.path.basename(file_path)}...", end=' ', flush=True)
        status_file = file_path + '.status'

        if os.path.exists(status_file):
            skipped_count += 1
            print("Skipped (already processed).")
            continue

        # --- Parse the file (outside the timed process) ---
        start_time = time.time()
        af_graph = parse_af_file(file_path)
        arguments_list = sorted(list(af_graph.nodes()))
        attacks_list = list(af_graph.edges())
        
        if not arguments_list:
            print("Error (empty or unreadable file).")
            error_count += 1
            with open(status_file, 'w') as f:
                f.write('error: parsing failed')
            continue

        # --- Run calculation in a separate process with timeout ---
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=calculate_dbs_worker,
            args=(result_queue, arguments_list, attacks_list, 0)
        )
        process.start()
        process.join(timeout=timeout_seconds)

        duration = time.time() - start_time

        # --- Handle the result ---
        if process.is_alive():
            process.terminate()
            process.join()
            timeout_count += 1
            timed_out_files.append(file_path)
            with open(status_file, 'w') as f:
                f.write('timeout')
            print(f"Timeout after {timeout_seconds}s.")
        else:
            try:
                result = result_queue.get_nowait()
                if isinstance(result, Exception):
                    error_count += 1
                    with open(status_file, 'w') as f:
                        f.write(f'error: {result}')
                    print(f"Error during calculation: {result}")
                else:
                    success_count += 1
                    with open(status_file, 'w') as f:
                        f.write('success')
                    print(f"Success ({duration:.2f}s).")
                    # Note: We don't write the result ranking to a file as per requirements.
            except queue.Empty:
                error_count += 1
                with open(status_file, 'w') as f:
                    f.write('error: process finished with no result')
                print("Error (process finished unexpectedly).")

    # --- 3. Print Final Summary ---
    print("\n\n--- Final Analysis Summary ---")
    print(f"Total Frameworks Found:    {total_files}")
    print(f"Successfully Processed:      {success_count}")
    print(f"Skipped (already done):      {skipped_count}")
    print(f"Failed (Timeout):            {timeout_count}")
    print(f"Failed (Other Error):        {error_count}")
    
    if timed_out_files:
        print("\n--- Frameworks that Timed Out ---")
        for f in timed_out_files:
            print(f"- {f}")


if __name__ == '__main__':
    run_full_analysis()
