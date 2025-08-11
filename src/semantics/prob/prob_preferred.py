import networkx as nx
from typing import List, FrozenSet, Any
from .prob_base import ProbabilisticSemantics
from .prob_complete import ProbComplete

class ProbPreferred(ProbabilisticSemantics):
    """
    Probabilistic ranking based on preferred semantics.
    Finds all preferred extensions by filtering the set of all complete extensions.
    """
    def _find_extensions_in_subgraph(self, subgraph: nx.DiGraph) -> List[FrozenSet[Any]]:
        """
        Finds all preferred extensions by first finding all complete extensions
        and then identifying the maximal ones.
        """
        if not subgraph.nodes:
            return [frozenset()]

        # Reuse the complete semantics solver to get all complete extensions
        complete_finder = ProbComplete(subgraph)
        complete_exts = complete_finder._find_extensions_in_subgraph(subgraph)

        if not complete_exts:
            return []

        # Filter for maximal sets (preferred extensions)
        preferred_extensions = []
        for s1 in complete_exts:
            is_maximal = True
            for s2 in complete_exts:
                if s1 != s2 and s1.issubset(s2):
                    is_maximal = False
                    break
            if is_maximal:
                preferred_extensions.append(s1)
                
        return preferred_extensions