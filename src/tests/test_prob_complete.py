# tests/test_prob_complete.py

import pytest
import networkx as nx
from semantics.prob.prob_complete import ProbComplete

# =============================================================================
# Unit Tests for the core extension-finding logic
# =============================================================================

def test_prob_complete_extensions_on_simple_attack(simple_attack_framework):
    """
    Unit test for _find_extensions_in_subgraph on a simple attack graph (1->2).
    The only complete extension is {'1'}.
    """
    prob = ProbComplete(simple_attack_framework)
    extensions = prob._find_extensions_in_subgraph(simple_attack_framework)
    expected_extensions = {frozenset({'1'})}
    assert set(extensions) == expected_extensions

def test_prob_complete_extensions_on_3_cycle(three_cycle_framework):
    """
    Unit test for _find_extensions_in_subgraph on a 3-cycle graph.
    The only complete extension is the empty set.
    """
    prob = ProbComplete(three_cycle_framework)
    extensions = prob._find_extensions_in_subgraph(three_cycle_framework)
    expected_extensions = {frozenset()}
    assert set(extensions) == expected_extensions

def test_prob_complete_extensions_on_mutual_attack(mutual_attack_framework):
    """
    Unit test for _find_extensions_in_subgraph on a mutual attack graph (1<->2).
    The complete extensions are {}, {1}, and {2}.
    """
    prob = ProbComplete(mutual_attack_framework)
    extensions = prob._find_extensions_in_subgraph(mutual_attack_framework)
    expected_extensions = {frozenset(), frozenset({'1'}), frozenset({'2'})}
    assert set(extensions) == expected_extensions

# =============================================================================
# Integration Tests for the final probabilistic scores
# =============================================================================

def test_prob_complete_scores_on_simple_attack(simple_attack_framework):
    """
    Tests the final probabilistic scores for a simple attack graph (1->2).
    """
    prob = ProbComplete(simple_attack_framework, p=0.5)
    scores = prob.get_scores()
    expected_scores = {'1': 0.5, '2': 0.25}
    assert scores == pytest.approx(expected_scores, rel=0.1)

def test_prob_complete_scores_on_3_cycle(three_cycle_framework):
    """
    Tests the final probabilistic scores for a 3-cycle graph.
    """
    prob = ProbComplete(three_cycle_framework, p=0.5)
    scores = prob.get_scores()
    expected_scores = {'1': 0.25, '2': 0.25, '3': 0.25}
    assert scores == pytest.approx(expected_scores, rel=0.1)

def test_prob_complete_scores_on_mutual_attack(mutual_attack_framework):
    """
    Tests the final probabilistic scores for a mutual attack graph (1<->2).
    """
    prob = ProbComplete(mutual_attack_framework, p=0.5)
    scores = prob.get_scores()
    expected_scores = {'1': 0.5, '2': 0.5}
    assert scores == pytest.approx(expected_scores, rel=0.1)