# semantics/dbs.py

import networkx as nx
import numpy as np
from scipy.sparse import csr_array
from typing import List, Dict, Set

class Dbs:
    def __init__(self, af: nx.DiGraph, max_path_length: int = 0):
        if not isinstance(af, nx.DiGraph):
            raise TypeError("Argumentation framework must be a NetworkX DiGraph.")

        self.af = af
        self.arguments = sorted(list(af.nodes))
        self.num_args = len(self.arguments)
        
        # If max_path_length is not specified (or 0), default to num_args
        self.max_path_length = max_path_length if max_path_length > 0 else self.num_args
        
        self._discussion_vectors: Dict[str, List[int]] = {}
        self._ranking: List[Set[str]] = []
        self._calculate_ranking()

    def _calculate_ranking(self):
        if self.num_args == 0:
            return

        adj_matrix_T = nx.to_scipy_sparse_array(
            self.af, nodelist=self.arguments, dtype=np.int64
        ).transpose().tocsr()
        
        self._discussion_vectors = {arg: [] for arg in self.arguments}
        
        current_power = adj_matrix_T.copy()
        
        for path_length in range(1, self.max_path_length + 1):
            num_paths_per_arg = np.array(current_power.sum(axis=1)).flatten()
            
            sign = 1 if path_length % 2 != 0 else -1
            
            for i, arg in enumerate(self.arguments):
                self._discussion_vectors[arg].append(sign * int(num_paths_per_arg[i]))
            
            if path_length < self.max_path_length:
                if current_power.nnz == 0:
                    # Pad the rest of the vectors with zeros
                    for arg in self.arguments:
                        self._discussion_vectors[arg].extend([0] * (self.max_path_length - path_length))
                    break
                current_power = current_power.dot(adj_matrix_T)

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

    def get_discussion_vectors(self) -> Dict[str, List[int]]:
        return self._discussion_vectors

    def get_ranking(self) -> List[Set[str]]:
        return self._ranking