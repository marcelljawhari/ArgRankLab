# tests/test_prob_admissible.py

import pytest
import networkx as nx
from semantics.prob.prob_admissible import ProbAdmissible

def test_prob_admissible_scores_on_AF_ex(af_ex_framework):
    """
    Tests the analytical scores on the AF_ex.af file.
    """
    prob = ProbAdmissible(af_ex_framework, p=0.5)
    calculated_scores = prob.get_scores()
    expected_scores = {
        '1': 0.5,
        '2': 0.25,
        '3': 0.125,
        '4': 0.125,
        '5': 0.25,
        '6': 0.5,
        '7': 0.25,
        '8': 0.125
    }
    assert calculated_scores == pytest.approx(expected_scores)

def test_prob_admissible_scores_on_single_argument(single_arg_framework):
    """
    Tests the scoring on a framework with a single, unattacked argument.
    """
    prob = ProbAdmissible(single_arg_framework, p=0.5)
    scores = prob.get_scores()
    expected = {'1': 0.5}
    assert scores == pytest.approx(expected)

def test_prob_admissible_scores_on_simple_attack(simple_attack_framework):
    """
    Tests the scoring on a framework with a simple attack (1 -> 2).
    """
    prob = ProbAdmissible(simple_attack_framework, p=0.5)
    scores = prob.get_scores()
    expected = {
        '1': 0.5,
        '2': 0.25
    }
    assert scores == pytest.approx(expected)

def test_prob_admissible_scores_on_self_attack(self_attack_framework):
    """
    Tests that a self-attacking argument has a score of 0.
    """
    prob = ProbAdmissible(self_attack_framework, p=0.5)
    scores = prob.get_scores()
    expected = {'1': 0.0}
    assert scores == pytest.approx(expected)

def test_prob_admissible_scores_on_mutual_attack(mutual_attack_framework):
    """
    Tests a mutual attack (1 <-> 2) where each argument can defend itself.
    """
    prob = ProbAdmissible(mutual_attack_framework, p=0.5)
    scores = prob.get_scores()
    expected = {
        '1': 0.5,
        '2': 0.5
    }
    assert scores == pytest.approx(expected)