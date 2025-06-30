# tests/test_prob_ideal.py

import pytest
import networkx as nx
from semantics.prob.prob_ideal import ProbIdeal

# =============================================================================
# Unit Tests for the Ideal Extension Logic
# =============================================================================

def test_prob_ideal_extensions_on_simple_attack(simple_attack_framework):
    """
    Tests that for 1->2, the ideal extension is {1}.
    """
    prob = ProbIdeal(simple_attack_framework)
    # Preferred Extensions: [{1}]. Intersection: {1}. Ideal: {1}.
    extensions = prob._find_extensions_in_subgraph(simple_attack_framework)
    expected_extensions = [frozenset({'1'})]
    assert extensions == expected_extensions

def test_prob_ideal_extensions_on_mutual_attack(mutual_attack_framework):
    """
    Tests that for 1<->2, the ideal extension is the empty set.
    """
    prob = ProbIdeal(mutual_attack_framework)
    # Preferred Extensions: [{1}, {2}]. Intersection: {}. Ideal: {}.
    extensions = prob._find_extensions_in_subgraph(mutual_attack_framework)
    expected_extensions = [frozenset()]
    assert extensions == expected_extensions

def test_prob_ideal_extensions_on_split_defense(split_defense_framework):
    """
    Tests a framework where the ideal extension is non-empty.
    """
    prob = ProbIdeal(split_defense_framework)
    # Corrected Logic: The only Preferred Extension is {1, 2, 4}.
    # Therefore, the intersection is {1, 2, 4}, and the Ideal Extension is {1, 2, 4}.
    extensions = prob._find_extensions_in_subgraph(split_defense_framework)
    expected_extensions = [frozenset({'1', '2', '4'})]
    assert extensions == expected_extensions

# =============================================================================
# Integration Tests for the Final Probabilistic Scores
# =============================================================================

def test_prob_ideal_scores_on_simple_attack(simple_attack_framework):
    """
    Tests the final probabilistic scores for a simple attack graph (1 -> 2).
    """
    prob = ProbIdeal(simple_attack_framework, p=0.5)
    scores = prob.get_scores()
    # Ideal is {1} in subgraphs {1} and {1,2}. Count=2. Score=2/4=0.5.
    # Ideal is {2} in subgraph {2}. Count=1. Score=1/4=0.25.
    expected_scores = {'1': 0.5, '2': 0.25}
    assert scores == pytest.approx(expected_scores, rel=0.1)

def test_prob_ideal_scores_on_mutual_attack(mutual_attack_framework):
    """
    Tests the final probabilistic scores for a mutual attack graph (1 <-> 2).
    """
    prob = ProbIdeal(mutual_attack_framework, p=0.5)
    scores = prob.get_scores()
    # Arg '1' is in Ideal Ext only in subgraph {1}. Count=1. Score=1/4=0.25.
    # Arg '2' is in Ideal Ext only in subgraph {2}. Count=1. Score=1/4=0.25.
    expected_scores = {'1': 0.25, '2': 0.25}
    assert scores == pytest.approx(expected_scores, rel=0.1)