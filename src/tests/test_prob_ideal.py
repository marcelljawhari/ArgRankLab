# tests/test_prob_ideal.py

import pytest
import networkx as nx
from semantics.prob.prob_ideal import ProbIdeal

def test_prob_ideal_extensions_on_simple_attack(simple_attack_framework):
    """
    Tests that for 1->2, the ideal extension is {1}.
    """
    prob = ProbIdeal(simple_attack_framework)
    extensions = prob._find_extensions_in_subgraph(simple_attack_framework)
    expected_extensions = [frozenset({'1'})]
    assert extensions == expected_extensions

def test_prob_ideal_extensions_on_mutual_attack(mutual_attack_framework):
    """
    Tests that for 1<->2, the ideal extension is the empty set.
    """
    prob = ProbIdeal(mutual_attack_framework)
    extensions = prob._find_extensions_in_subgraph(mutual_attack_framework)
    expected_extensions = [frozenset()]
    assert extensions == expected_extensions

def test_prob_ideal_extensions_on_split_defense(split_defense_framework):
    """
    Tests a framework where the ideal extension is non-empty.
    """
    prob = ProbIdeal(split_defense_framework)
    extensions = prob._find_extensions_in_subgraph(split_defense_framework)
    expected_extensions = [frozenset({'1', '2', '4'})]
    assert extensions == expected_extensions

def test_prob_ideal_scores_on_simple_attack(simple_attack_framework):
    """
    Tests the final probabilistic scores for a simple attack graph (1 -> 2).
    """
    prob = ProbIdeal(simple_attack_framework, p=0.5)
    scores = prob.get_scores()
    expected_scores = {'1': 0.5, '2': 0.25}
    assert scores == pytest.approx(expected_scores, rel=0.1)

def test_prob_ideal_scores_on_mutual_attack(mutual_attack_framework):
    """
    Tests the final probabilistic scores for a mutual attack graph (1 <-> 2).
    """
    prob = ProbIdeal(mutual_attack_framework, p=0.5)
    scores = prob.get_scores()
    expected_scores = {'1': 0.25, '2': 0.25}
    assert scores == pytest.approx(expected_scores, rel=0.1)