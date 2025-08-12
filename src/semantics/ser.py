# semantics/ser.py

import os
import time
from typing import List, Set, FrozenSet, Dict, Any
from pysat.solvers import Glucose4
import networkx as nx

class Ser:
    """
    Implements the serialisability-based ranking semantics using a SAT-based
    approach to find initial sets, as described by Bengel & Thimm (2023).
    """
    def __init__(self, af: nx.DiGraph, max_recursion_depth: int = 15):
        if not isinstance(af, nx.DiGraph):
            raise TypeError("Argumentation framework must be a NetworkX DiGraph.")

        self.af = af
        self.original_arguments = set(af.nodes)
        self.max_depth = max_recursion_depth
        
        self._serialisation_indices: Dict[Any, float] = {arg: float('inf') for arg in self.original_arguments}
        self._ranking: List[Set[Any]] = []

        self._calculate_ranking_hybrid_pruned()

    def _calculate_ranking_hybrid_pruned(self):
        """Calculates serialisation indices using the SAT-powered hybrid approach."""
        initial_sets_step1 = self._find_initial_sets_sat(self.af)
        if not initial_sets_step1:
            return

        for s in initial_sets_step1:
            for arg in s:
                self._serialisation_indices[arg] = 1
        
        for s_i in initial_sets_step1:
            self._explore_sequences_pruned(frozenset(s_i), 2)
        
        sorted_args = sorted(list(self.original_arguments), key=lambda arg: (self._serialisation_indices[arg], str(arg)))
        if sorted_args:
            from itertools import groupby
            for key, group in groupby(sorted_args, key=lambda arg: self._serialisation_indices[arg]):
                self._ranking.append(set(group))

    def _explore_sequences_pruned(self, accepted_set: FrozenSet[Any], step: int):
        """Recursively explores sequences, pruning branches that cannot improve results."""
        if step > self.max_depth: return
        attacked_by_accepted = self._get_attacked_set(accepted_set)
        nodes_for_reduct = self.original_arguments - accepted_set - attacked_by_accepted

        can_improve = any(self._serialisation_indices[arg] > step for arg in nodes_for_reduct)
        if not can_improve: return

        current_subgraph = self.af.subgraph(nodes_for_reduct)
        initial_sets = self._find_initial_sets_sat(current_subgraph)
        if not initial_sets: return

        for s_i in initial_sets:
            for arg in s_i:
                self._serialisation_indices[arg] = min(self._serialisation_indices[arg], step)
            new_accepted_set = accepted_set.union(s_i)
            self._explore_sequences_pruned(new_accepted_set, step + 1)

    def _get_attacked_set(self, source_set: Set[Any]) -> Set[Any]:
        attacked = set()
        for arg in source_set: attacked.update(self.af.successors(arg))
        return attacked
        
    def _find_initial_sets_sat(self, graph: nx.DiGraph) -> List[FrozenSet[Any]]:
        """
        Finds all initial sets in a graph using a SAT solver.
        This function encodes the properties of a minimal non-empty admissible set
        into a SAT formula and iteratively queries a solver.
        """
        if not graph.nodes:
            return []

        nodes = sorted(list(graph.nodes()))
        arg_map = {arg: i + 1 for i, arg in enumerate(nodes)}
        
        initial_sets = []
        
        # Base clauses for being an admissible, non-empty set
        base_clauses = self._encode_admissible(graph, arg_map)
        base_clauses.append([v for v in arg_map.values()]) # Must be non-empty

        with Glucose4(bootstrap_with=base_clauses) as solver:
            while solver.solve():
                model = solver.get_model()
                current_admissible_set = frozenset(nodes[i] for i, v in enumerate(model) if v > 0 and v < len(nodes) + 1)
                
                # Check for minimality by seeing if a proper subset is also admissible.
                is_minimal = True
                if len(current_admissible_set) > 1:
                    # Create a new, temporary problem to search for a smaller set.
                    # It inherits the base rules of admissibility.
                    with Glucose4(bootstrap_with=base_clauses) as min_check_solver:
                        # Constraint: The solution must be a SUBSET of the current set.
                        # For every argument NOT in our current set, it cannot be in the solution.
                        for node in nodes:
                            if node not in current_admissible_set:
                                min_check_solver.add_clause([-arg_map[node]])
                        
                        # Constraint: The solution must be a PROPER subset.
                        # At least one argument FROM our current set must be excluded.
                        min_check_solver.add_clause([-arg_map[arg] for arg in current_admissible_set])
                        
                        if min_check_solver.solve():
                            # A smaller admissible set exists, so the current one is not minimal.
                            is_minimal = False

                if is_minimal:
                    initial_sets.append(current_admissible_set)
                    # Block this minimal model from being found again.
                    blocking_clause = [-arg_map[arg] for arg in current_admissible_set]
                    solver.add_clause(blocking_clause)
                else:
                    # The found set was not minimal. Block it and force the solver to
                    # find a different (possibly smaller) solution in the next iteration.
                    blocking_clause = [-arg_map[arg] if arg in current_admissible_set else arg_map[arg] for arg in nodes]
                    # This clause is complex, a simpler way is to just block the found model
                    blocking_clause_simple = [-v for v in model if v > 0 and v <= len(nodes)]
                    solver.add_clause(blocking_clause_simple)

        return initial_sets


    def _encode_admissible(self, graph: nx.DiGraph, arg_map: Dict[Any, int]) -> List[List[int]]:
        """
        Translates the properties of admissibility into a CNF formula for the SAT solver.
        Variables are positive for 'in' the set.
        """
        clauses = []
        
        # 1. Conflict-Freeness: If 'a' attacks 'b', 'a' and 'b' cannot both be in.
        for a, b in graph.edges:
            if a in arg_map and b in arg_map:
                clauses.append([-arg_map[a], -arg_map[b]])

        # 2. Admissibility (Defense): If an argument 'a' is in, it must be defended.
        for a in graph.nodes:
            if a not in arg_map: continue
            attackers = list(graph.predecessors(a))
            if not attackers: continue

            for b in attackers:
                defenders = list(graph.predecessors(b))
                # Clause: (¬I_a) ∨ (I_c1 ∨ I_c2 ∨ ...)
                # If 'a' is in, at least one of its defenders against 'b' must be in.
                clause = [-arg_map[a]] + [arg_map[c] for c in defenders if c in arg_map]
                clauses.append(clause)
        
        return clauses

    def get_serialisation_indices(self) -> Dict[str, float]:
        return self._serialisation_indices

    def get_ranking(self) -> List[Set[Any]]:
        return self._ranking
