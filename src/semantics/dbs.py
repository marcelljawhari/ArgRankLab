import networkx as nx
import numpy as np
from itertools import groupby
from typing import List

def calculate_dbs_ranking(graph: nx.DiGraph, max_depth: int = 10) -> List[List[str]]:
    """
    Calculates the ranking of arguments using Discussion-based Semantics (Dbs).

    The ranking is based on the lexicographical comparison of discussion vectors,
    which count attack and defense paths of different lengths. This implementation
    uses matrix exponentiation to find the number of paths up to max_depth.

    Args:
        graph: A networkx.DiGraph representing the argumentation framework.
        max_depth: The maximum length of attack sequences to consider for the vectors.

    Returns:
        A list of lists, where each inner list is an equivalence class of
        equally ranked arguments, ordered from most to least acceptable.
    """
    if not graph.nodes():
        return []

    # Get a sorted list of nodes to ensure consistent matrix-to-node mapping
    sorted_nodes = sorted(list(graph.nodes()))
    node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}

    # Create the adjacency matrix M, where M[i, j] = 1 if i attacks j
    adj_matrix = nx.to_numpy_array(graph, nodelist=sorted_nodes, dtype=int)
    
    discussion_vectors = {node: [] for node in sorted_nodes}
    
    # M_i will hold the i-th power of the adjacency matrix
    m_i = adj_matrix.copy()

    for i in range(1, max_depth + 1):
        # The number of paths of length 'i' ending at node 'j' is the sum
        # of the j-th column of the matrix M^i.
        path_counts = np.sum(m_i, axis=0)
        
        for node in sorted_nodes:
            idx = node_to_idx[node]
            num_paths = path_counts[idx]
            
            # Odd length sequences are attacks (+), even are defenses (-)
            score = num_paths if i % 2 != 0 else -num_paths
            discussion_vectors[node].append(score)

        # Calculate the next power of the matrix for the next iteration
        if i < max_depth:
            m_i = m_i @ adj_matrix

    # Sort arguments based on their discussion vectors.
    # The Dbs ranking prefers smaller vectors lexicographically.
    # Python's default sort on lists/tuples is lexicographical, which is exactly what we need.
    # We first sort all arguments...
    all_args_sorted = sorted(sorted_nodes, key=lambda arg: discussion_vectors[arg])
    
    # ...then group them by identical vectors to form equivalence classes.
    final_ranking = [
        list(group) for key, group in groupby(all_args_sorted, key=lambda arg: discussion_vectors[arg])
    ]

    return final_ranking