# semantics/dbs.py

import networkx as nx
import numpy as np
import scipy.sparse
import time # Optional: for timing the process

class Dbs:
    """
    Implements the Discussion-based ranking semantics (Dbs).
    """
    def __init__(self, af: nx.DiGraph, max_path_length: int = 10):
        if not isinstance(af, nx.DiGraph):
            raise TypeError("Argumentation framework must be a NetworkX DiGraph.")

        self.af = af
        self.arguments = sorted(list(af.nodes))
        self.arg_to_index = {arg: i for i, arg in enumerate(self.arguments)}
        self.num_args = len(self.arguments)
        self.max_path_length = max_path_length
        
        self._discussion_vectors = {}
        self._ranking = []

        self._calculate_all_optimized()

    def _calculate_all_optimized(self):
        """
        A method that first pre-computes all required matrix
        powers, then constructs the discussion vectors in a second, fast pass.
        """
        print("Optimized Dbs calculation started...")
        start_time = time.time()

        # --- Stage 1: Pre-computation of Matrix Powers ---
        # We use the transpose of the adjacency matrix. The number of paths of
        # length k ENDING at node 'j' in the original graph is equal to the
        # number of paths of length k STARTING from node 'j' in the transpose.
        # This allows us to sum ROWS, which is faster with the 'csr' format.
        adj_matrix_T = nx.to_scipy_sparse_array(
            self.af, nodelist=self.arguments
        ).transpose().tocsr()

        matrix_powers = []
        current_power = adj_matrix_T.copy()

        print(f"Pre-computing {self.max_path_length} matrix powers for {self.num_args} arguments...")
        for _ in range(self.max_path_length):
            matrix_powers.append(current_power)
            current_power = current_power @ adj_matrix_T
        
        print(f"Matrix power calculation finished in {time.time() - start_time:.2f} seconds.")

        # --- Stage 2: Fast Construction of Discussion Vectors ---
        # Initialize a dictionary to hold the vectors
        scores = {arg: [0] * self.max_path_length for arg in self.arguments}
        
        # Iterate through the pre-computed powers
        for i, m_power in enumerate(matrix_powers):
            path_length = i + 1
            # Get the number of paths for ALL arguments at this path length at once
            # by summing the rows of the transposed matrix power.
            num_paths_vector = m_power.sum(axis=1).flatten() # .flatten() is the modern way

            value = num_paths_vector if path_length % 2 != 0 else -num_paths_vector

            # Assign the calculated scores for this path length to all arguments
            for arg_idx, arg_name in enumerate(self.arguments):
                scores[arg_name][i] = int(value[arg_idx])

        self._discussion_vectors = scores
        print(f"Discussion vector construction finished in {time.time() - start_time:.2f} seconds.")

        # --- Stage 3: Sorting (same as before) ---
        sorted_args = sorted(self.arguments, key=lambda arg: self._discussion_vectors[arg])
        
        if not sorted_args:
            return

        self._ranking = []
        current_rank_group = {sorted_args[0]}
        for i in range(1, len(sorted_args)):
            prev_arg = sorted_args[i-1]
            curr_arg = sorted_args[i]
            
            if self._discussion_vectors[curr_arg] == self._discussion_vectors[prev_arg]:
                current_rank_group.add(curr_arg)
            else:
                self._ranking.append(current_rank_group)
                current_rank_group = {curr_arg}
        
        self._ranking.append(current_rank_group)
        print(f"Total calculation time: {time.time() - start_time:.2f} seconds.")

    def get_discussion_vectors(self) -> dict[str, list[int]]:
        """Returns the calculated discussion count vector for each argument."""
        return self._discussion_vectors

    def get_ranking(self) -> list[set[str]]:
        """Returns the final ranking of arguments from most to least acceptable."""
        return self._ranking