# semantics/prob/prob_ideal.py

import networkx as nx
from .prob_base import ProbabilisticSemantics
from .prob_preferred import ProbPreferred # Ideal semantics depends on preferred
from typing import List, FrozenSet, Any

class ProbIdeal(ProbabilisticSemantics):
    """Probabilistic ranking based on ideal semantics."""

    def _find_extensions_in_subgraph(self, subgraph: nx.DiGraph) -> List[FrozenSet[Any]]:
        """
        Finds the unique ideal extension.
        It is the largest admissible set contained in the intersection of all
        preferred extensions.
        """
        if not subgraph.nodes:
            return []

        # Step 1: Find all preferred extensions
        preferred_finder = ProbPreferred(subgraph)
        preferred_extensions = preferred_finder._find_extensions_in_subgraph(subgraph)

        if not preferred_extensions:
            # Should not happen if graph is non-empty, as empty set is admissible
            return [frozenset()]

        # Step 2: Find the intersection of all preferred extensions
        intersection_of_preferred = set.intersection(*map(set, preferred_extensions))

        if not intersection_of_preferred:
            return [frozenset()]

        # Step 3: Find the largest admissible subset of this intersection
        ideal_extension = frozenset()
        
        # Iterate through powerset of the intersection to find admissible sets
        intersection_nodes = list(intersection_of_preferred)
        for i in range(1 << len(intersection_nodes)):
            subset = frozenset(node for j, node in enumerate(intersection_nodes) if (i >> j) & 1)
            
            if len(subset) > len(ideal_extension) and self._is_admissible(subgraph, subset):
                ideal_extension = subset
                
        return [ideal_extension]

    def _is_admissible(self, subgraph: nx.DiGraph, s: FrozenSet[Any]) -> bool:
        """Checks if a set is admissible."""
        if not self._is_conflict_free(subgraph, s):
            return False
        for arg in s:
            if not self._defends(subgraph, s, arg):
                return False
        return True