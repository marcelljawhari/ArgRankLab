import networkx as nx
import numpy as np
from collections import defaultdict

class Dbs:
    """
    Implements the Discussion-based ranking semantics (Dbs) for an
    abstract argumentation framework.

    This corrected version uses matrix exponentiation of the adjacency matrix
    to accurately count all attack paths (including non-simple ones with cycles),
    aligning with the formal definition and the thesis example.
    """

    def __init__(self, af: nx.DiGraph, max_path_length: int = 10):
        """
        Initializes the DbsRanking calculator.

        Args:
            af (nx.DiGraph): The argumentation framework represented as a
                             NetworkX directed graph. Nodes are arguments, and
                             a directed edge (u, v) means 'u attacks v'.
            max_path_length (int): The maximum length of attack sequences to
                                   consider. This is necessary to ensure termination.
                                   Defaults to 10.
        """
        if not isinstance(af, nx.DiGraph):
            raise TypeError("Argumentation framework must be a NetworkX DiGraph.")

        self.af = af
        # Ensure a fixed order for matrix operations
        self.arguments = sorted(list(af.nodes))
        self.arg_to_index = {arg: i for i, arg in enumerate(self.arguments)}
        self.max_path_length = max_path_length
        
        self._discussion_vectors = {}
        self._ranking = []

        self._calculate_all()

    def _calculate_discussion_vector(self, argument: str) -> list[int]:
        """
        Calculates the discussion count vector for a single argument using
        the adjacency matrix to count all paths.

        Dis(a) = <Dis_1(a), Dis_2(a), ..., Dis_k(a)>
        where Dis_i is based on the number of attack paths of length i (edges).

        Args:
            argument (str): The argument to calculate the vector for.

        Returns:
            The discussion count vector as a list of integers.
        """
        # Get the adjacency matrix of the graph.
        # Note: In NetworkX/Numpy adjacency matrix M[i, j] = 1 means an edge from i to j.
        # An attack path a_k -> ... -> a_1 -> x is a path from a_k to x.
        # So we need the standard adjacency matrix.
        adj_matrix = nx.to_numpy_array(self.af, nodelist=self.arguments)

        dis_vector = [0] * self.max_path_length
        arg_index = self.arg_to_index[argument]
        
        # M_power will hold the i-th power of the adjacency matrix
        m_power = np.copy(adj_matrix)

        for i in range(1, self.max_path_length + 1):
            # The number of paths of length 'i' from any node to 'argument'
            # is the sum of the column corresponding to 'argument' in M^i.
            num_paths = int(np.sum(m_power[:, arg_index]))
            
            # Dis_i is positive for odd length (attack) and negative for even (defense)
            if i % 2 != 0:
                dis_vector[i-1] = num_paths
            else:
                dis_vector[i-1] = -num_paths

            # Prepare for the next iteration: M^(i+1) = M^i * M
            if i < self.max_path_length:
                m_power = m_power @ adj_matrix
                
        return dis_vector

    def _calculate_all(self):
        """
        Calculates the discussion vectors for all arguments and the final ranking.
        """
        scores = {}
        for arg in self.arguments:
            scores[arg] = self._calculate_discussion_vector(arg)
        self._discussion_vectors = scores

        sorted_args = sorted(self.arguments, key=lambda arg: scores[arg])
        
        self._ranking = []
        if not sorted_args:
            return

        current_rank_group = {sorted_args[0]}
        for i in range(1, len(sorted_args)):
            prev_arg = sorted_args[i-1]
            curr_arg = sorted_args[i]
            
            if scores[curr_arg] == scores[prev_arg]:
                current_rank_group.add(curr_arg)
            else:
                self._ranking.append(current_rank_group)
                current_rank_group = {curr_arg}
        
        self._ranking.append(current_rank_group)

    def get_discussion_vectors(self) -> dict[str, list[int]]:
        """
        Returns the calculated discussion count vector for each argument.
        """
        return self._discussion_vectors

    def get_ranking(self) -> list[set[str]]:
        """
        Returns the final ranking of arguments from most to least acceptable.
        """
        return self._ranking