# semantics/prob/prob_grounded.py

import networkx as nx
from .prob_base import ProbabilisticSemantics
from typing import List, FrozenSet, Any

class ProbGrounded(ProbabilisticSemantics):
    """
    Probabilistic ranking based on the grounded semantics.
    Uses an efficient iterative algorithm to find the unique grounded extension.
    """
    def _find_extensions_in_subgraph(self, subgraph: nx.DiGraph) -> List[FrozenSet[Any]]:
        """
        Calculates the unique grounded extension of the given subgraph.
        The algorithm iteratively adds arguments that are defended by the
        current set of accepted arguments, starting with unattacked arguments.
        """
        if not subgraph.nodes:
            return []

        accepted_args = set()
        while True:
            newly_accepted_args = set()
            for node in subgraph.nodes:
                if node in accepted_args:
                    continue

                # An argument is accepted if all its attackers are defeated by the current set
                is_defended = True
                attackers = list(subgraph.predecessors(node))
                if not attackers: # Unattacked arguments are in by default
                    newly_accepted_args.add(node)
                    continue

                for attacker in attackers:
                    # Check if an argument in the current accepted set attacks this attacker
                    is_counter_attacked = False
                    for defender in accepted_args:
                        if subgraph.has_edge(defender, attacker):
                            is_counter_attacked = True
                            break
                    if not is_counter_attacked:
                        is_defended = False
                        break
                
                if is_defended:
                    newly_accepted_args.add(node)

            if newly_accepted_args.issubset(accepted_args):
                # No new arguments were accepted in this iteration, fixpoint reached.
                break
            
            accepted_args.update(newly_accepted_args)

        return [frozenset(accepted_args)] if accepted_args else []