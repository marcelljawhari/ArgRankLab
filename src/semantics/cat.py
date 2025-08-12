# semantics/cat.py

import networkx as nx
import numpy as np
import scipy.sparse
import time

class Cat:
    """
    Implements the Categoriser-based ranking semantics (Cat).
    """
    def __init__(self, af: nx.DiGraph, tolerance: float = 1e-8, max_iterations: int = 1000):
        """
        Initializes the Categoriser semantics calculator.

        Args:
            af: The argumentation framework as a NetworkX DiGraph.
            tolerance: The convergence threshold for the iterative calculation.
                       The process stops when the maximum change in any argument's
                       strength is less than this value.
            max_iterations: The maximum number of iterations to perform before
                            stopping, even if convergence is not reached.
        """
        if not isinstance(af, nx.DiGraph):
            raise TypeError("Argumentation framework must be a NetworkX DiGraph.")

        self.af = af
        self.arguments = sorted(list(af.nodes))
        self.arg_to_index = {arg: i for i, arg in enumerate(self.arguments)}
        self.num_args = len(self.arguments)
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        
        self._strengths = {}
        self._ranking = []

        self._calculate_strengths()
        self._build_ranking()

    def _calculate_strengths(self):
        """
        Calculates the strength of each argument using a vectorized iterative method.

        The strength of an argument 'a' is defined as:
        - 1, if 'a' has no attackers.
        - 1 / (1 + sum(strength(b))) for all attackers 'b' of 'a'.

        This method solves this system of equations for all arguments simultaneously.
        """
        # print("Categoriser-based semantics calculation started...")
        start_time = time.time()

        if self.num_args == 0:
            # print("Graph has no arguments.")
            return

        # We need the transpose of the adjacency matrix.
        # adj_T[i, j] = 1 means argument `j` attacks argument `i`.
        # This allows us to compute the sum of attacker strengths for all
        # arguments at once using a single matrix-vector product.
        adj_T = nx.to_scipy_sparse_array(
            self.af, nodelist=self.arguments
        ).transpose().tocsr()

        # Initialize strengths vector S with zeros.
        strengths_vector = np.zeros(self.num_args)

        # print(f"Iteratively solving for {self.num_args} arguments...")
        for i in range(self.max_iterations):
            old_strengths = strengths_vector

            # Calculate the sum of strengths of all attackers for each argument.
            attacker_strengths_sum = adj_T @ old_strengths

            # Update the strength of every argument in a single vectorized operation.
            strengths_vector = 1 / (1 + attacker_strengths_sum)
            
            # Check for convergence using the infinity norm (maximum absolute difference).
            if np.linalg.norm(strengths_vector - old_strengths, ord=np.inf) < self.tolerance:
                # print(f"Converged after {i + 1} iterations.")
                break
        else: # This else clause executes if the for loop completes without a 'break'.
            print(f"Warning: Did not converge within {self.max_iterations} iterations.")

        self._strengths = {arg: strengths_vector[self.arg_to_index[arg]] for arg in self.arguments}
        # print(f"Strength calculation finished in {time.time() - start_time:.2f} seconds.")

    def _build_ranking(self):
        """
        Sorts arguments into a ranking based on their calculated strengths.
        Arguments with nearly identical strengths are grouped together.
        """
        if not self._strengths:
            return

        # Sort arguments by strength in descending order (higher strength is better).
        sorted_args = sorted(self.arguments, key=lambda arg: self._strengths[arg], reverse=True)
        
        self._ranking = []
        current_rank_group = {sorted_args[0]}

        for i in range(1, len(sorted_args)):
            prev_arg = sorted_args[i-1]
            curr_arg = sorted_args[i]
            
            # Group arguments if their strengths are within the tolerance margin.
            if abs(self._strengths[curr_arg] - self._strengths[prev_arg]) < self.tolerance:
                current_rank_group.add(curr_arg)
            else:
                self._ranking.append(current_rank_group)
                current_rank_group = {curr_arg}
        
        self._ranking.append(current_rank_group)

    def get_strengths(self) -> dict[str, float]:
        """Returns the calculated strength for each argument."""
        return self._strengths

    def get_ranking(self) -> list[set[str]]:
        """Returns the final ranking of arguments from most to least acceptable."""
        return self._ranking