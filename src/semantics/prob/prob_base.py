# semantics/prob/prob_base.py

import abc
import time
import random
from collections import defaultdict
from typing import Any, Dict, FrozenSet, List, Set

import networkx

class ProbabilisticSemantics(abc.ABC):
    """
    An abstract base class for all probabilistic ranking semantics.
    (Full docstring omitted for brevity)
    """
    def __init__(self,
                 af: networkx.DiGraph,
                 num_samples: int = 10000,
                 p: float = 0.5):
        """
        Args:
            af: The argumentation framework.
            num_samples: The number of Monte Carlo iterations.
            p: The uniform probability of an argument's existence.
        """
        self.af = af
        self.num_samples = num_samples
        self.p = p
        self.all_nodes: List[Any] = list(self.af.nodes)
        
        self._scores: Dict[Any, float] | None = None
        self._ranking: List[Set[Any]] | None = None

    @abc.abstractmethod
    def _find_extensions_in_subgraph(self, subgraph: networkx.DiGraph) -> List[FrozenSet[Any]]:
        """
        Finds all extensions in a given subgraph for a specific semantics.
        This is the contract that all concrete semantics classes must fulfill.
        """
        raise NotImplementedError

    def get_scores(self) -> Dict[str, float]:
        """Returns a dictionary mapping each argument to its probabilistic score."""
        if self._scores is None:
            self._calculate_scores()
        return self._scores

    def get_ranking(self) -> List[Set[str]]:
        """Returns the final ranking of arguments based on their scores."""
        if self._ranking is None:
            scores = self.get_scores()
            grouped_by_score = defaultdict(set)
            for arg, score in scores.items():
                found_group = False
                for s_group in grouped_by_score.keys():
                    if abs(score - s_group) < 1e-9:
                        grouped_by_score[s_group].add(arg)
                        found_group = True
                        break
                if not found_group:
                    grouped_by_score[score].add(arg)
            
            sorted_scores = sorted(grouped_by_score.keys(), reverse=True)
            self._ranking = [grouped_by_score[score] for score in sorted_scores]
        return self._ranking

    def _calculate_scores(self) -> None:
        """
        Performs the Monte Carlo simulation.
        NOTE: This method is now designed to be called by a parallel worker,
        so it processes a chunk of samples, not the whole set.
        """
        # This method is left for potential single-threaded use,
        # but the primary execution is now handled by the parallel main.py.
        # The logic is kept here for reference.

        acceptance_counts = defaultdict(int)
        use_fixed_size_heuristic = len(self.all_nodes) > 30 and self.__class__.__name__ != 'ProbGrounded'

        for i in range(self.num_samples):
            print(f"\rProcessing sample {i+1}/{self.num_samples}...", end="")

            if use_fixed_size_heuristic:
                # ==========================================================
                # THE CHANGE IS HERE: Sample size reduced from 20 to 16
                # ==========================================================
                sample_size = min(16, len(self.all_nodes))
                subgraph_nodes = random.sample(self.all_nodes, sample_size)
            else:
                subgraph_nodes = [node for node in self.all_nodes if random.random() < self.p]

            subgraph = self.af.subgraph(subgraph_nodes)

            if not subgraph.nodes:
                continue

            extensions = self._find_extensions_in_subgraph(subgraph)

            if not extensions:
                continue

            credulously_accepted = frozenset.union(*extensions)
            for arg in credulously_accepted:
                if arg in self.all_nodes:
                    acceptance_counts[arg] += 1

        print("\rProcessing complete.               ")
        self._scores = {node: acceptance_counts[node] / self.num_samples for node in self.all_nodes}

    def _is_conflict_free(self, graph: networkx.DiGraph, subset: FrozenSet[Any]) -> bool:
        """Checks if a given subset of arguments is conflict-free."""
        for arg1 in subset:
            for arg2 in subset:
                if graph.has_edge(arg1, arg2):
                    return False
        return True

    def _defends(self, graph: networkx.DiGraph, subset: FrozenSet[Any], argument: Any) -> bool:
        """Checks if a subset defends a given argument."""
        for attacker in graph.predecessors(argument):
            is_defended = False
            for defender in subset:
                if graph.has_edge(defender, attacker):
                    is_defended = True
                    break
            if not is_defended:
                return False
        return True