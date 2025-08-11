# classify_frameworks.py

import os
import networkx as nx
import pandas as pd
from tqdm import tqdm

from util.af_parser import parse_af_file

# --- Configuration ---
BENCHMARK_DIRS = [
    os.path.join("data", "benchmarks_tweety"),
    os.path.join("data", "benchmarks2023", "main")
]
RESULTS_DIR = os.path.join("data", "results")
OUTPUT_CSV = os.path.join("data", "framework_properties.csv")

# --- Helper Functions ---

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

def get_framework_properties(af: nx.DiGraph) -> dict:
    """Analyzes an AF and returns a dictionary of its structural properties."""
    num_nodes = af.number_of_nodes()
    if num_nodes == 0: return {}

    properties = {}
    
    # 1. Cyclicity
    properties['cyclicity'] = "Cyclic" if not nx.is_directed_acyclic_graph(af) else "Acyclic"
    
    # 2. Size
    if num_nodes < 25: properties['size_group'] = "Small"
    elif 25 <= num_nodes <= 75: properties['size_group'] = "Medium"
    else: properties['size_group'] = "Large"

    # 3. Density
    num_edges = af.number_of_edges()
    max_edges = num_nodes * (num_nodes - 1)
    density = num_edges / max_edges if max_edges > 0 else 0
    properties['density_value'] = density
    if density < 0.05: properties['density_group'] = "Sparse"
    elif 0.05 <= density <= 0.15: properties['density_group'] = "Medium"
    else: properties['density_group'] = "Dense"

    # 4. Connectivity
    num_components = nx.number_weakly_connected_components(af)
    properties['connectivity'] = "Connected" if num_components == 1 else "Disconnected"
    properties['num_components'] = num_components
    
    return properties

# --- Main Execution Logic ---

def main():
    """
    Main function to classify all AFs, check their processing status,
    and save the combined metadata.
    """
    print("=" * 60)
    print("  Framework Classifier & Status Checker  ")
    print("=" * 60)

    framework_paths = find_framework_files(BENCHMARK_DIRS)
    if not framework_paths:
        print("\nError: No '.af' files found in any benchmark directories.")
        return

    print(f"\nFound {len(framework_paths)} frameworks to classify and check...")
    
    all_properties_data = []
    for af_path in tqdm(framework_paths, desc="Classifying and Checking Status"):
        try:
            # --- Get Structural Properties ---
            af = parse_af_file(af_path)
            properties = get_framework_properties(af)
            
            if not properties: continue

            base_name = os.path.basename(af_path)
            record = {
                'framework_name': base_name,
                'source_dataset': "tweety" if "benchmarks_tweety" in af_path else "iccma23",
                'num_args': af.number_of_nodes(),
                'num_attacks': af.number_of_edges(),
                **properties
            }
            
            # --- Check Processing Status ---
            status = "Not Processed" # Default status
            base_name_no_ext = base_name.replace('.af', '')
            
            timeout_marker_path = os.path.join(RESULTS_DIR, f"{base_name_no_ext}.timeout")
            kendall_path = os.path.join(RESULTS_DIR, f"{base_name_no_ext}_kendall.csv")
            spearman_path = os.path.join(RESULTS_DIR, f"{base_name_no_ext}_spearman.csv")
            
            if os.path.exists(timeout_marker_path):
                status = "Timed Out"
            elif os.path.exists(kendall_path) and os.path.exists(spearman_path):
                status = "Processed"
            
            record['status'] = status
            all_properties_data.append(record)

        except Exception as e:
            print(f"Could not process {os.path.basename(af_path)}: {e}")
    
    # --- Save the combined data ---
    properties_df = pd.DataFrame(all_properties_data)
    properties_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nClassification complete.")
    print(f"Saved metadata for {len(properties_df)} frameworks to '{OUTPUT_CSV}'")
    print("\n--- Status Summary ---")
    print(properties_df['status'].value_counts().to_string())
    print("=" * 60)

if __name__ == '__main__':
    # Optional: Install tqdm for a nice progress bar: pip install tqdm
    try:
        from tqdm import tqdm
    except ImportError:
        print("Hint: Install 'tqdm' (pip install tqdm) for a progress bar.")
        def tqdm(iterable, **kwargs):
            return iterable
            
    main()