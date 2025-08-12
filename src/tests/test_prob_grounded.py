# tests/test_prob_grounded.py

import pytest
import networkx as nx
from semantics.prob.prob_grounded import ProbGrounded

def test_prob_grounded_extensions_on_simple_attack(simple_attack_framework):
    """
    Tests that for 1->2, the grounded extension is {1}.
    """
    prob = ProbGrounded(simple_attack_framework)
    extensions = prob._find_extensions_in_subgraph(simple_attack_framework)
    expected_extensions = [frozenset({'1'})]
    assert extensions == expected_extensions

def test_prob_grounded_extensions_on_3_cycle(three_cycle_framework):
    """
    Tests that for a 3-cycle, the grounded extension is the empty set.
    """
    prob = ProbGrounded(three_cycle_framework)
    extensions = prob._find_extensions_in_subgraph(three_cycle_framework)
    assert extensions == [frozenset()]

def test_prob_grounded_extensions_on_defense_chain(defense_chain_framework):
    """
    Tests that for 1->2->3, the grounded extension is {1, 3}.
    """
    prob = ProbGrounded(defense_chain_framework)
    extensions = prob._find_extensions_in_subgraph(defense_chain_framework)
    expected_extensions = [frozenset({'1', '3'})]
    assert extensions == expected_extensions

def test_prob_grounded_scores_on_simple_attack(simple_attack_framework):
    """
    Tests the final probabilistic scores for a simple attack graph (1 -> 2).
    """
    prob = ProbGrounded(simple_attack_framework, p=0.5)
    scores = prob.get_scores()
    expected_scores = {'1': 0.5, '2': 0.25}
    assert scores == pytest.approx(expected_scores, rel=0.1)

def test_prob_grounded_scores_on_3_cycle(three_cycle_framework):
    """
    Tests the final probabilistic scores for a 3-cycle graph.
    """
    prob = ProbGrounded(three_cycle_framework, p=0.5)
    scores = prob.get_scores()
    expected_scores = {'1': 0.25, '2': 0.25, '3': 0.25}
    assert scores == pytest.approx(expected_scores, rel=0.1)

def test_prob_grounded_scores_on_AF_ex(af_ex_framework):
    """
    Tests the final probabilistic scores on the AF_ex framework.
    """
    prob = ProbGrounded(af_ex_framework, p=0.5)
    scores = prob.get_scores()

    expected_scores = {
        '1': 128 / 256,  # 0.5
        '2': 64 / 256,   # 0.25
        '3': 48 / 256,   # 0.1875
        '4': 32 / 256,   # 0.125
        '5': 64 / 256,   # 0.25
        '6': 128 / 256,  # 0.5
        '7': 80 / 256,   # 0.3125
        '8': 80 / 256    # 0.3125
    }
    assert scores == pytest.approx(expected_scores, rel=0.1)