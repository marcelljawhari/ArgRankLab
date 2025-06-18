# semantics/prob/prob_complete.py

import networkx as nx
from .prob_base import ProbabilisticSemantics
from typing import List, FrozenSet, Any, Set

class ProbComplete(ProbabilisticSemantics):
    """
    Probabilistic ranking based on complete semantics.
    This version uses an efficient two-step process:
    1. Find all admissible sets using recursive backtracking.
    2. Filter the admissible sets to find which are complete.
    """

    def _find_extensions_in_subgraph(self, subgraph: nx.DiGraph) -> List[FrozenSet[Any]]:
        """Finds all complete extensions using an optimized method."""
        nodes = list(subgraph.nodes)
        if not nodes:
            return []

        # Step 1: Find all admissible sets efficiently.
        admissible_sets = self._find_admissible_recursively(subgraph, nodes)
        
        complete_extensions = []
        # Step 2: From the admissible sets, find which are complete.
        for s in admissible_sets:
            # A set 's' is complete if it contains exactly the arguments it defends.
            defended_by_s = set()
            for node in nodes:
                if self._defends(subgraph, s, node):
                    defended_by_s.add(node)
            
            if s == defended_by_s:
                complete_extensions.append(s)
                
        return complete_extensions

    def _find_admissible_recursively(self, subgraph: nx.DiGraph, nodes: List[Any]) -> List[FrozenSet[Any]]:
        """
        Uses recursive backtracking to find all admissible sets.
        """
        admissible_sets = []

        def find_and_check(candidate_set: Set[Any], remaining_nodes: List[Any], depth: int):
            # Check if the current candidate is admissible.
            # It's conflict-free by construction, so we only need to check if it defends itself.
            is_admissible = all(self._defends(subgraph, candidate_set, arg) for arg in candidate_set)
            
            if is_admissible:
                admissible_sets.append(frozenset(candidate_set))

            # Explore further candidates by adding one node at a time
            for i in range(len(remaining_nodes)):
                new_node = remaining_nodes[i]
                
                # Pruning step: Check for conflicts with the new node before recursing
                is_conflicted = any(
                    subgraph.has_edge(new_node, existing_node) or 
                    subgraph.has_edge(existing_node, new_node) 
                    for existing_node in candidate_set
                )

                if not is_conflicted:
                    candidate_set.add(new_node)
                    find_and_check(candidate_set, remaining_nodes[i+1:], depth + 1)
                    candidate_set.remove(new_node) # Backtrack

        find_and_check(set(), nodes, 0)
        return admissible_sets