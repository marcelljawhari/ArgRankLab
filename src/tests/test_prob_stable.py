# tests/test_prob_stable.py

import pytest
import networkx as nx
import math
from semantics.prob.prob_stable import ProbStable

def test_prob_stable_scores_on_AF_ex(af_ex_framework):
    """
    Tests the analytical log-scores on the main AF_ex example.
    The score is log(p) + (num_nodes - 1 - out_degree) * log(1-p).
    """
    prob = ProbStable(af_ex_framework, p=0.5)
    scores = prob.get_scores()

    log_p = math.log(0.5)
    num_nodes = af_ex_framework.number_of_nodes()

    expected_scores = {
        '1': log_p * (num_nodes - 3),
        '2': log_p * (num_nodes - 1),
        '3': log_p * (num_nodes - 0),
        '4': log_p * (num_nodes - 1),
        '5': log_p * (num_nodes - 1),
        '6': log_p * (num_nodes - 1),
        '7': log_p * (num_nodes - 1),
        '8': log_p * (num_nodes - 1)
    }

    assert scores == pytest.approx(expected_scores)

def test_prob_stable_scores_on_simple_attack(simple_attack_framework):
    """
    Tests log-scores for a simple attack graph (1 -> 2).
    """
    prob = ProbStable(simple_attack_framework, p=0.5)
    scores = prob.get_scores()
    log_p = math.log(0.5)

    expected_scores = {'1': log_p, '2': 2 * log_p}
    assert scores == pytest.approx(expected_scores)

def test_prob_stable_scores_on_mutual_attack(mutual_attack_framework):
    """
    Tests log-scores for a mutual attack graph (1 <-> 2).
    """
    prob = ProbStable(mutual_attack_framework, p=0.5)
    scores = prob.get_scores()
    log_p = math.log(0.5)

    expected_scores = {'1': log_p, '2': log_p}
    assert scores == pytest.approx(expected_scores)

def test_prob_stable_scores_on_self_attack(self_attack_framework):
    """
    Tests that a self-attacking argument has a log-score of -inf.
    """
    prob = ProbStable(self_attack_framework, p=0.5)
    scores = prob.get_scores()
    expected_scores = {'1': -math.inf}
    assert scores == expected_scores

def test_prob_stable_scores_on_single_argument(single_arg_framework):
    """
    Tests the log-score for a single, unattacked argument.
    """
    prob = ProbStable(single_arg_framework, p=0.5)
    scores = prob.get_scores()
    log_p = math.log(0.5)

    expected_scores = {'1': log_p}
    assert scores == pytest.approx(expected_scores)