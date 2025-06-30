# tests/test_prob_preferred.py

import pytest
import networkx as nx
from semantics.prob.prob_preferred import ProbPreferred

# =============================================================================
# Unit Tests for the Preferred Extension Logic
# =============================================================================

def test_prob_preferred_extensions_on_simple_attack(simple_attack_framework):
    """
    Tests that for 1->2, the only preferred extension is {1}.
    """
    prob = ProbPreferred(simple_attack_framework)
    # Admissible sets are {} and {1}. {1} is maximal.
    extensions = prob._find_extensions_in_subgraph(simple_attack_framework)
    expected_extensions = {frozenset({'1'})}
    assert set(extensions) == expected_extensions

def test_prob_preferred_extensions_on_mutual_attack(mutual_attack_framework):
    """
    Tests that for 1<->2, the preferred extensions are {1} and {2}.
    """
    prob = ProbPreferred(mutual_attack_framework)
    # Admissible sets are {}, {1}, {2}. {1} and {2} are maximal.
    extensions = prob._find_extensions_in_subgraph(mutual_attack_framework)
    expected_extensions = {frozenset({'1'}), frozenset({'2'})}
    assert set(extensions) == expected_extensions

def test_prob_preferred_extensions_on_3_cycle(three_cycle_framework):
    """
    Tests that for a 3-cycle, the only preferred extension is the empty set.
    """
    prob = ProbPreferred(three_cycle_framework)
    # The only admissible set is {}, which is therefore maximal.
    extensions = prob._find_extensions_in_subgraph(three_cycle_framework)
    expected_extensions = {frozenset()}
    assert set(extensions) == expected_extensions

# =============================================================================
# Integration Tests for the Final Probabilistic Scores
# =============================================================================

def test_prob_preferred_scores_on_simple_attack(simple_attack_framework):
    """
    Tests the final probabilistic scores for a simple attack graph (1 -> 2).
    """
    prob = ProbPreferred(simple_attack_framework, p=0.5)
    scores = prob.get_scores()
    # Arg '1' in Pref Ext in subgraphs {1}, {1,2}. Count=2. Score=2/4=0.5.
    # Arg '2' in Pref Ext in subgraph {2}. Count=1. Score=1/4=0.25.
    expected_scores = {'1': 0.5, '2': 0.25}
    assert scores == pytest.approx(expected_scores, rel=0.1)

def test_prob_preferred_scores_on_mutual_attack(mutual_attack_framework):
    """
    Tests the final probabilistic scores for a mutual attack graph (1 <-> 2).
    """
    prob = ProbPreferred(mutual_attack_framework, p=0.5)
    scores = prob.get_scores()
    # Arg '1' in Pref Ext in subgraphs {1}, {1,2}. Count=2. Score=2/4=0.5.
    # Arg '2' in Pref Ext in subgraphs {2}, {1,2}. Count=2. Score=2/4=0.5.
    expected_scores = {'1': 0.5, '2': 0.5}
    assert scores == pytest.approx(expected_scores, rel=0.1)

def test_prob_preferred_scores_on_3_cycle(three_cycle_framework):
    """
    Tests the final probabilistic scores for a 3-cycle graph.
    """
    prob = ProbPreferred(three_cycle_framework, p=0.5)
    scores = prob.get_scores()
    # Arg '1' in Pref Ext in subgraphs {1}, {1,2}. Count=2. Score=2/8=0.25.
    expected_scores = {'1': 0.25, '2': 0.25, '3': 0.25}
    assert scores == pytest.approx(expected_scores, rel=0.1)