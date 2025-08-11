import networkx as nx
from typing import List, FrozenSet, Any
from .prob_base import ProbabilisticSemantics

try:
    import pysat.solvers
except ImportError:
    pysat = None

class ProbComplete(ProbabilisticSemantics):
    """
    Probabilistic ranking based on complete semantics.
    Uses a state-of-the-art SAT-based algorithm to find all complete extensions.
    """
    def _find_extensions_in_subgraph(self, subgraph: nx.DiGraph) -> List[FrozenSet[Any]]:
        """
        Finds all complete extensions using an Exhaustive Extension Enumeration (EEE)
        approach with a SAT solver.
        """
        if not subgraph.nodes:
            return [frozenset()]
        if pysat is None:
            raise ImportError("PySAT library is required for this method. Please install with 'pip install python-sat'")
            
        clauses, arg_map, var_map = self._get_complete_encoding(subgraph)
        N = len(var_map)
        
        extensions = []
        with pysat.solvers.Glucose4() as solver:
            solver.append_formula(clauses)
            while solver.solve():
                model = solver.get_model()
                # Extract the extension from the 'in' variables of the model
                ext = frozenset({arg_map[v] for v in model if v > 0 and v <= N})
                extensions.append(ext)
                # Add a blocking clause to find a different model in the next iteration
                solver.add_clause([-v for v in model if abs(v) <= N])
                
        return extensions