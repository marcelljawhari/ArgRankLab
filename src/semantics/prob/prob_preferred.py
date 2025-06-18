# semantics/prob/prob_preferred.py

import networkx as nx
from .prob_base import ProbabilisticSemantics
from typing import List, FrozenSet, Any, Set

class ProbPreferred(ProbabilisticSemantics):
    """
    Probabilistic ranking based on preferred semantics.
    This version uses an efficient recursive backtracking search.
    """

    def _find_extensions_in_subgraph(self, subgraph: nx.DiGraph) -> List[FrozenSet[Any]]:
        """
        Finds all preferred extensions, which are maximal admissible sets.
        """
        nodes = list(subgraph.nodes)
        
        # Step 1: Find all admissible sets using an efficient recursive search.
        admissible_sets = self._find_admissible_recursively(subgraph, nodes)

        # Step 2: Filter the admissible sets to find the maximal ones.
        preferred_extensions = []
        for s1 in admissible_sets:
            is_maximal = True
            for s2 in admissible_sets:
                # Check if s1 is a strict subset of s2
                if s1 != s2 and s1.issubset(s2):
                    is_maximal = False
                    break
            if is_maximal:
                preferred_extensions.append(s1)
                
        # If no non-empty admissible sets exist, the empty set is the only preferred one.
        if not preferred_extensions and frozenset() in admissible_sets:
            return [frozenset()]

        return preferred_extensions

    def _find_admissible_recursively(self, subgraph: nx.DiGraph, nodes: List[Any]) -> List[FrozenSet[Any]]:
        """
        Uses recursive backtracking to find all admissible sets without a full
        powerset check, which is much more efficient.
        """
        admissible_sets = []

        def find_and_check(candidate_set: Set[Any], remaining_nodes: List[Any], depth: int):
            # First, check if the current candidate set is admissible.
            is_admissible_so_far = True
            for arg in candidate_set:
                if not self._defends(subgraph, candidate_set, arg):
                    is_admissible_so_far = False
                    break
            
            if is_admissible_so_far:
                admissible_sets.append(frozenset(candidate_set))

            # Pruning: If the set is not admissible, no superset can be,
            # so we don't need to explore further down this path.
            # This check is implicitly handled by the logic above. If a set
            # is not self-defending, we don't stop the search for *other*
            # branches, but we won't add its supersets from this path.

            # Explore further candidates by adding one node at a time
            for i in range(len(remaining_nodes)):
                new_node = remaining_nodes[i]
                
                # Check for conflicts only with the new node to be added
                is_conflicted = any(
                    subgraph.has_edge(new_node, existing_node) or 
                    subgraph.has_edge(existing_node, new_node) 
                    for existing_node in candidate_set
                )

                if not is_conflicted:
                    candidate_set.add(new_node)
                    find_and_check(candidate_set, remaining_nodes[i+1:], depth + 1)
                    candidate_set.remove(new_node) # Backtrack

        # Start the recursive search with an empty set
        find_and_check(set(), nodes, 0)
        return admissible_sets