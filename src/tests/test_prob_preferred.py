# tests/test_prob_preferred.py

import pytest
import networkx as nx
from semantics.prob.prob_preferred import ProbPreferred

def test_prob_preferred_extensions_on_simple_attack(simple_attack_framework):
    """
    Tests that for 1->2, the only preferred extension is {1}.
    """
    prob = ProbPreferred(simple_attack_framework)
    extensions = prob._find_extensions_in_subgraph(simple_attack_framework)
    expected_extensions = {frozenset({'1'})}
    assert set(extensions) == expected_extensions

def test_prob_preferred_extensions_on_mutual_attack(mutual_attack_framework):
    """
    Tests that for 1<->2, the preferred extensions are {1} and {2}.
    """
    prob = ProbPreferred(mutual_attack_framework)
    extensions = prob._find_extensions_in_subgraph(mutual_attack_framework)
    expected_extensions = {frozenset({'1'}), frozenset({'2'})}
    assert set(extensions) == expected_extensions

def test_prob_preferred_extensions_on_3_cycle(three_cycle_framework):
    """
    Tests that for a 3-cycle, the only preferred extension is the empty set.
    """
    prob = ProbPreferred(three_cycle_framework)
    extensions = prob._find_extensions_in_subgraph(three_cycle_framework)
    expected_extensions = {frozenset()}
    assert set(extensions) == expected_extensions

def test_prob_preferred_scores_on_simple_attack(simple_attack_framework):
    """
    Tests the final probabilistic scores for a simple attack graph (1 -> 2).
    """
    prob = ProbPreferred(simple_attack_framework, p=0.5)
    scores = prob.get_scores()
    expected_scores = {'1': 0.5, '2': 0.25}
    assert scores == pytest.approx(expected_scores, rel=0.1)

def test_prob_preferred_scores_on_mutual_attack(mutual_attack_framework):
    """
    Tests the final probabilistic scores for a mutual attack graph (1 <-> 2).
    """
    prob = ProbPreferred(mutual_attack_framework, p=0.5)
    scores = prob.get_scores()
    expected_scores = {'1': 0.5, '2': 0.5}
    assert scores == pytest.approx(expected_scores, rel=0.1)

def test_prob_preferred_scores_on_3_cycle(three_cycle_framework):
    """
    Tests the final probabilistic scores for a 3-cycle graph.
    """
    prob = ProbPreferred(three_cycle_framework, p=0.5)
    scores = prob.get_scores()
    expected_scores = {'1': 0.25, '2': 0.25, '3': 0.25}
    assert scores == pytest.approx(expected_scores, rel=0.1)