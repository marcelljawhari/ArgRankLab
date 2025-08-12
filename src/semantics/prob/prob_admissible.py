# semantics/prob_admissible.py

import networkx as nx
from collections import defaultdict

class ProbAdmissible:
    """
    Calculates a probabilistic score for arguments based on the analytical
    probability that the singleton set {arg} is admissible.
    This avoids Monte Carlo simulation, leveraging the PTIME complexity
    of admissible semantics as shown by Fazzinga et al. (2014).
    """
    def __init__(self, af: nx.DiGraph, p: float = 0.5):
        self.af = af
        self.p = p # Uniform probability of existence for each argument
        self._scores = None

    def get_scores(self) -> dict[str, float]:
        """Calculates and returns the scores for all arguments."""
        if self._scores is None:
            self._calculate_scores()
        return self._scores

    def _calculate_scores(self) -> None:
        """
        Computes the score for each argument 'a' as Pr({a} is admissible).
        """
        self._scores = {}
        all_args = list(self.af.nodes)

        for i, arg_a in enumerate(all_args):
            # print(f"\rCalculating analytical score for argument {i+1}/{len(all_args)}...", end="")

            # 1. Probability that 'a' itself exists.
            prob_a_exists = self.p

            # 2. Probability that 'a' is conflict-free (i.e., no self-attack).
            # If a self-attack (a,a) exists, it must not be chosen. In the problem model,
            # attacks only exist if both nodes exist, so Pr(self_attack) = p.
            # We assume attacks in the base graph are certain if nodes exist.
            prob_a_is_cf = 1.0 if not self.af.has_edge(arg_a, arg_a) else 0.0

            # 3. Probability that 'a' is defended against all attackers.
            # This is a product over all potential attackers 'b'.
            prob_a_is_defended = 1.0
            attackers_of_a = list(self.af.predecessors(arg_a))

            for arg_b in attackers_of_a:
                if arg_b == arg_a: continue # Handled by conflict-free check
                
                prob_b_is_counter_attacked = (1 - self.p)
                if self.af.has_edge(arg_a, arg_b):
                    prob_b_is_counter_attacked += self.p

                prob_a_is_defended *= prob_b_is_counter_attacked

            # Final score for 'a' is the product of these probabilities.
            self._scores[arg_a] = prob_a_exists * prob_a_is_cf * prob_a_is_defended
        
        # print("\rAnalytical calculation complete.                      ")