# ===================================================================
# File: semantics/prob/prob_base.py
# Note: This is now a smart, dispatching base class.
# ===================================================================
import abc
import random
from collections import defaultdict
from typing import Any, Dict, FrozenSet, List, Set
import networkx as nx

try:
    import pysat.solvers
except ImportError:
    pysat = None

class ProbabilisticSemantics(abc.ABC):
    def __init__(self, af: nx.DiGraph, num_samples: int = 10000, p: float = 0.5):
        self.af = af
        self.num_samples = num_samples
        self.p = p
        self.all_nodes: List[Any] = list(self.af.nodes)
        self._scores: Dict[Any, float] | None = None
        # REMOVED: The fixed exact_threshold is no longer needed.

    @abc.abstractmethod
    def _find_extensions_in_subgraph(self, subgraph: nx.DiGraph) -> List[FrozenSet[Any]]:
        raise NotImplementedError

    def get_scores(self) -> Dict[str, float]:
        if self._scores is None:
            self._calculate_scores()
        return self._scores

    def _get_all_subgraphs(self):
        """Generator that yields all 2^n possible non-empty subgraphs."""
        nodes = self.all_nodes
        num_nodes = len(nodes)
        # Iterate from 1 to 2^n - 1 to skip the empty set, which has no accepted arguments.
        for i in range(1, 1 << num_nodes):
            subgraph_nodes = [nodes[j] for j in range(num_nodes) if (i >> j) & 1]
            yield self.af.subgraph(subgraph_nodes)

    def _calculate_scores(self) -> None:
        """
        Smart dispatcher for calculating probabilistic scores.
        - If the total number of possible subgraphs is less than the number of
          samples, it performs an exact calculation.
        - Otherwise, it falls back to a Monte Carlo simulation.
        """
        num_nodes = len(self.all_nodes)
        
        # --- Dynamic Decision Logic ---
        # Calculate the total number of possible subgraphs (2^n).
        # The (1 << num_nodes) is a fast way to compute 2**num_nodes.
        total_combinations = 1 << num_nodes
        
        # --- Exact Calculation Path ---
        if total_combinations < self.num_samples:
            self._scores = {node: 0.0 for node in self.all_nodes}
            
            for subgraph in self._get_all_subgraphs():
                num_in = subgraph.number_of_nodes()
                num_out = num_nodes - num_in
                subgraph_prob = (self.p ** num_in) * ((1 - self.p) ** num_out)

                extensions = self._find_extensions_in_subgraph(subgraph)
                credulously_accepted = frozenset.union(*extensions) if extensions else frozenset()

                for arg in credulously_accepted:
                    self._scores[arg] += subgraph_prob
            return

        # --- Monte Carlo Simulation Path ---
        acceptance_counts = defaultdict(int)
        for _ in range(self.num_samples):
            subgraph_nodes = [node for node in self.all_nodes if random.random() < self.p]
            if not subgraph_nodes:
                continue
            subgraph = self.af.subgraph(subgraph_nodes)
            extensions = self._find_extensions_in_subgraph(subgraph)
            credulously_accepted = frozenset.union(*extensions) if extensions else frozenset()
            for arg in credulously_accepted:
                acceptance_counts[arg] += 1
        self._scores = {node: acceptance_counts.get(node, 0) / self.num_samples for node in self.all_nodes}

    # Helper for SAT-based semantics
    def _get_complete_encoding(self, subgraph: nx.DiGraph):
        nodes = list(subgraph.nodes)
        arg_map = {i + 1: arg for i, arg in enumerate(nodes)}
        var_map = {arg: i + 1 for i, arg in enumerate(nodes)}
        N = len(nodes)
        clauses = []
        for a in nodes:
            in_a, out_a, undec_a = var_map[a], var_map[a] + N, var_map[a] + 2 * N
            clauses.extend([[in_a, out_a, undec_a], [-in_a, -out_a], [-in_a, -undec_a], [-out_a, -undec_a]])
            attackers_in = [var_map[b] for b in subgraph.predecessors(a)]
            if attackers_in:
                clauses.append([-out_a] + attackers_in)
                for att_var in attackers_in: clauses.append([out_a, -att_var])
            else:
                clauses.append([-out_a])
            attackers_out = [var_map[b] + N for b in subgraph.predecessors(a)]
            if attackers_out:
                clauses.append([in_a] + [-v for v in attackers_out])
                for v in attackers_out: clauses.append([-in_a, v])
            else:
                clauses.append([in_a])
        return clauses, arg_map, var_map