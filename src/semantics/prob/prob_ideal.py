import networkx as nx
from typing import List, FrozenSet, Any, Set, Optional
from .prob_base import ProbabilisticSemantics

try:
    import pysat.solvers
except ImportError:
    pysat = None

class ProbIdeal(ProbabilisticSemantics):
    """
    Probabilistic ranking based on ideal semantics.
    Uses the state-of-the-art CDIS algorithm to find the unique ideal extension
    without enumerating all preferred extensions.
    """
    def _find_admissible_attacker_of(self, subgraph: nx.DiGraph, candidate_set_P: Set[Any]) -> Optional[Set[Any]]:
        """
        Finds an admissible set S that attacks at least one argument in the candidate set P.
        """
        if not candidate_set_P:
            return None
        
        clauses, arg_map, var_map = self._get_complete_encoding(subgraph)
        N = len(var_map)

        with pysat.solvers.Glucose4() as solver:
            solver.append_formula(clauses)
            potential_attacker_vars = {var_map[s] for p in candidate_set_P for s in subgraph.predecessors(p) if s in var_map}
            
            if not potential_attacker_vars:
                return None

            solver.add_clause(list(potential_attacker_vars))
            
            if solver.solve():
                model = solver.get_model()
                return {arg_map[v] for v in model if v > 0 and v <= N}
        return None

    def _find_extensions_in_subgraph(self, subgraph: nx.DiGraph) -> List[FrozenSet[Any]]:
        """
        Finds the unique ideal extension using the CDIS algorithm.
        """
        if not subgraph.nodes:
            return [frozenset()]
        if pysat is None:
            raise ImportError("PySAT library is required for this method. Please install with 'pip install python-sat'")

        # Phase 1: Compute the Preferred Super-Core (P)
        P = set(subgraph.nodes)
        while True:
            S = self._find_admissible_attacker_of(subgraph, P)
            if not S:
                break
            
            attacked_by_S = {target for s_arg in S for target in subgraph.successors(s_arg)}
            P -= attacked_by_S
        
        # Phase 2: Compute the largest admissible set within P
        p_subgraph = subgraph.subgraph(P).copy()
        while True:
            removed_in_iteration = {
                p_arg for p_arg in p_subgraph.nodes() if not 
                all(any(p_subgraph.has_edge(d, att) for d in p_subgraph.nodes()) for att in p_subgraph.predecessors(p_arg))
            }
            
            if not removed_in_iteration:
                break
            
            p_subgraph.remove_nodes_from(removed_in_iteration)

        return [frozenset(p_subgraph.nodes())]