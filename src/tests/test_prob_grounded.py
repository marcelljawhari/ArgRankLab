# tests/test_prob_grounded.py

import pytest
import networkx as nx
from semantics.prob.prob_grounded import ProbGrounded

# =============================================================================
# Unit Tests for the Grounded Extension Logic
# =============================================================================

def test_prob_grounded_extensions_on_simple_attack(simple_attack_framework):
    """
    Tests that for 1->2, the grounded extension is {1}.
    """
    prob = ProbGrounded(simple_attack_framework)
    # The grounded extension contains the unattacked argument '1'.
    extensions = prob._find_extensions_in_subgraph(simple_attack_framework)
    expected_extensions = [frozenset({'1'})]
    assert extensions == expected_extensions

def test_prob_grounded_extensions_on_3_cycle(three_cycle_framework):
    """
    Tests that for a 3-cycle, the grounded extension is the empty set.
    """
    prob = ProbGrounded(three_cycle_framework)
    # With no unattacked arguments, the grounded extension is empty.
    extensions = prob._find_extensions_in_subgraph(three_cycle_framework)
    # The implementation returns an empty list for an empty extension.
    assert extensions == []

def test_prob_grounded_extensions_on_defense_chain(defense_chain_framework):
    """
    Tests that for 1->2->3, the grounded extension is {1, 3}.
    """
    prob = ProbGrounded(defense_chain_framework)
    # '1' is in (unattacked), which defends '3' by attacking '2'.
    extensions = prob._find_extensions_in_subgraph(defense_chain_framework)
    expected_extensions = [frozenset({'1', '3'})]
    assert extensions == expected_extensions

# =============================================================================
# Integration Tests for the Final Probabilistic Scores
# =============================================================================

def test_prob_grounded_scores_on_simple_attack(simple_attack_framework):
    """
    Tests the final probabilistic scores for a simple attack graph (1 -> 2).
    """
    prob = ProbGrounded(simple_attack_framework, p=0.5)
    scores = prob.get_scores()
    # Arg '1' is in GE in subgraphs {1} and {1,2}. Count=2. Score=2/4=0.5.
    # Arg '2' is in GE in subgraph {2}. Count=1. Score=1/4=0.25.
    expected_scores = {'1': 0.5, '2': 0.25}
    assert scores == pytest.approx(expected_scores, rel=0.1)

def test_prob_grounded_scores_on_3_cycle(three_cycle_framework):
    """
    Tests the final probabilistic scores for a 3-cycle graph.
    """
    prob = ProbGrounded(three_cycle_framework, p=0.5)
    scores = prob.get_scores()
    # By symmetry, all arguments have the same score.
    # Arg '1' is in GE in subgraphs {1} and {1,2}. Count=2. Score=2/8=0.25.
    expected_scores = {'1': 0.25, '2': 0.25, '3': 0.25}
    assert scores == pytest.approx(expected_scores, rel=0.1)