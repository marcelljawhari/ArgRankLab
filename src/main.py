# main.py (High-performance and memory-safe parallel version)

import os
import time
import math
from collections import defaultdict
from functools import partial

import networkx as nx
from util.af_parser import parse_af_file
import multiprocessing as mp
import random # Use standard random inside the worker

# Import all the concrete implementation classes
from semantics.prob.prob_admissible import ProbAdmissible
from semantics.prob.prob_stable import ProbStable
from semantics.prob.prob_grounded import ProbGrounded
from semantics.prob.prob_complete import ProbComplete
from semantics.prob.prob_preferred import ProbPreferred
from semantics.prob.prob_ideal import ProbIdeal

# A mapping from semantics name to the corresponding class
SEMANTICS_MAP_ANALYTICAL = {
    "admissible": ProbAdmissible,
    "stable": ProbStable
}

SEMANTICS_MAP_MC = {
    "grounded": ProbGrounded,
    "complete": ProbComplete,
    "preferred": ProbPreferred,
    "ideal": ProbIdeal
}

SEMANTICS_SAMPLE_COUNTS = {
    'default': 10000,
    'slow': 1250 
}

# --- (Worker functions init_worker and worker_run_single_sample remain unchanged) ---
worker_af = None
def init_worker(af_to_init: nx.DiGraph):
    global worker_af
    worker_af = af_to_init

def worker_run_single_sample(sample_index: int, SemanticsClass: type, p_val: float) -> set:
    calculator = SemanticsClass(af=worker_af, p=p_val)
    use_fixed_size_heuristic = len(calculator.all_nodes) > 30
    if use_fixed_size_heuristic:
        sample_size = min(16, len(calculator.all_nodes))
        subgraph_nodes = random.sample(calculator.all_nodes, sample_size)
    else:
        subgraph_nodes = [node for node in calculator.all_nodes if random.random() < calculator.p]
    subgraph = worker_af.subgraph(subgraph_nodes)
    if not subgraph.nodes: return set()
    extensions = calculator._find_extensions_in_subgraph(subgraph)
    if not extensions: return set()
    return set.union(*map(set, extensions))
# --- (End of worker functions) ---


def run_prob_for_all_semantics(framework_path: str, p_val: float = 0.5):
    if not os.path.exists(framework_path):
        print(f"Error: Framework file not found at {framework_path}")
        return

    print("="*60)
    print(f"Loading framework from: {framework_path}")
    print("="*60)
    arg_framework = parse_af_file(framework_path)
    print(f"Graph loaded with {arg_framework.number_of_nodes()} nodes and {arg_framework.number_of_edges()} edges.\n")

    # ======================= THE CHANGE IS HERE =======================
    # Limit the number of cores to prevent system instability on complex tasks.
    # Using half the cores is a safe starting point.
    available_cores = mp.cpu_count()
    num_cores = max(1, available_cores // 2)
    print(f"Using a safe limit of {num_cores} out of {available_cores} available CPU cores.")
    # ==================================================================

    all_semantics_names = list(SEMANTICS_MAP_ANALYTICAL.keys()) + list(SEMANTICS_MAP_MC.keys())

    for name in all_semantics_names:
        print(f"\n--- Calculating Probabilistic Ranking for '{name}' Semantics ---")
        start_time = time.time()
        
        scores = {}

        if name in SEMANTICS_MAP_ANALYTICAL:
            print("Using fast analytical method.")
            calculator = SEMANTICS_MAP_ANALYTICAL[name](af=arg_framework, p=p_val) 
            scores = calculator.get_scores()
        
        elif name in SEMANTICS_MAP_MC:
            if name in ['complete', 'preferred', 'ideal']:
                num_samples = SEMANTICS_SAMPLE_COUNTS['slow']
            else:
                num_samples = SEMANTICS_SAMPLE_COUNTS['default']

            print(f"Using Monte Carlo simulation with {num_samples} samples on {num_cores} cores.")
            
            SemanticsClass = SEMANTICS_MAP_MC[name]
            worker_func = partial(worker_run_single_sample, SemanticsClass=SemanticsClass, p_val=p_val)
            
            acceptance_counts = defaultdict(int)
            completed_count = 0
            
            with mp.Pool(processes=num_cores, initializer=init_worker, initargs=(arg_framework,)) as pool:
                results_iterator = pool.imap_unordered(worker_func, range(num_samples))

                for credulously_accepted_set in results_iterator:
                    for arg in credulously_accepted_set:
                        acceptance_counts[arg] += 1
                    
                    completed_count += 1
                    print(f"\rSimulations analyzed: {completed_count}/{num_samples}", end="")
            
            print() 

            scores = {arg: count / num_samples for arg, count in acceptance_counts.items()}
        
        end_time = time.time()
        print(f"Calculation finished in {end_time - start_time:.2f} seconds.")

        sorted_args = sorted(scores, key=scores.get, reverse=True)
        print("\nTop 5 Ranked Arguments:")
        for arg in sorted_args[:5]:
            format_str = "{:.4e}" if name == 'stable' and scores.get(arg, 0.0) < 0 else "{:.4f}"
            print(f"  Score({arg}) = {format_str.format(scores.get(arg, 0.0))}")
        
        print("\n" + "-"*60 + "\n")

if __name__ == '__main__':
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        print("Fork start method not available, using default.")

    af_file_path = os.path.join("data/benchmarks2023/main", "admbuster_2500000.af")
    run_prob_for_all_semantics(af_file_path)