import networkx as nx
from typing import List, FrozenSet, Any
from .prob_base import ProbabilisticSemantics

class ProbGrounded(ProbabilisticSemantics):
    """
    Probabilistic ranking based on the grounded semantics.
    Uses an efficient iterative algorithm to find the unique grounded extension.
    This algorithm is correct and efficient for all graph types.
    """
    def _find_extensions_in_subgraph(self, subgraph: nx.DiGraph) -> List[FrozenSet[Any]]:
        """
        Calculates the unique grounded extension of the given subgraph.
        """
        if not subgraph.nodes:
            return [frozenset()]

        accepted_args = set()
        while True:
            # Find all arguments that are defended by the current accepted set.
            newly_accepted = {
                n for n in subgraph.nodes if n not in accepted_args and
                all(any(subgraph.has_edge(d, att) for d in accepted_args) for att in subgraph.predecessors(n))
            }
            
            # If no new arguments were accepted, we have reached the fixpoint.
            if not newly_accepted:
                break
                
            accepted_args.update(newly_accepted)
            
        return [frozenset(accepted_args)]