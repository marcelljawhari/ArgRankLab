# semantics/prob/prob_stable.py

import networkx as nx
import math

class ProbStable:
    """
    Calculates probabilistic scores for arguments based on stable semantics.

    ============================================================================
    IMPORTANT IMPLEMENTATION NOTE:
    This implementation uses a fast, DIRECT ANALYTICAL method. To handle
    the extremely small probabilities resulting from the global nature of
    stable semantics on sparse graphs, it computes LOG-PROBABILITIES for
    ranking. A higher (less negative) score is better.
    ============================================================================
    """
    def __init__(self, af: nx.DiGraph, p: float = 0.5):
        self.af = af
        self.p = p
        self.all_nodes = list(self.af.nodes)
        self.num_nodes = len(self.all_nodes)
        self._scores = None

    def get_scores(self) -> dict[str, float]:
        """Calculates and returns the log-probability scores for all arguments."""
        if self._scores is None:
            self._calculate_scores()
        return self._scores

    def _calculate_scores(self) -> None:
        """
        Computes the log-score for each argument 'a' as log(Pr({a} is stable)).
        This is optimized to be O(N).
        """
        self._scores = {}
        
        # Pre-calculate logs for efficiency
        log_p = math.log(self.p)
        log_1_minus_p = math.log(1 - self.p)

        for i, arg_a in enumerate(self.all_nodes):
            # print(f"\rCalculating analytical log-score for argument {i+1}/{self.num_nodes}...", end="")

            # 1. Log-probability that 'a' itself exists.
            log_prob_a_exists = log_p

            # 2. Check for conflict-freeness. If conflicting, log-prob is -infinity.
            if self.af.has_edge(arg_a, arg_a):
                self._scores[arg_a] = -math.inf
                continue

            # 3. Log-probability that {a} attacks every other existing argument 'd'.
            # This is the sum of log-probabilities over all d != a.
            # log(Pr) = num_non_attacked * log(1-p)
            out_degree_a = self.af.out_degree(arg_a)
            num_non_attacked = (self.num_nodes - 1) - out_degree_a
            
            log_prob_attacks_external = num_non_attacked * log_1_minus_p
            
            # The final log-score is the sum of the log-probabilities.
            self._scores[arg_a] = log_prob_a_exists + log_prob_attacks_external
        
        # print("\rAnalytical calculation complete.                      ")