# tests/test_prob_admissible.py

import pytest
import networkx as nx
from semantics.prob.prob_admissible import ProbAdmissible

@pytest.fixture
def self_attack_framework():
    """Creates a framework with a single self-attacking argument."""
    graph = nx.DiGraph()
    graph.add_edge("1", "1")
    return graph

@pytest.fixture
def mutual_attack_framework():
    """Creates a framework with two mutually attacking arguments."""
    graph = nx.DiGraph()
    graph.add_edge("1", "2")
    graph.add_edge("2", "1")
    return graph

def test_prob_admissible_scores_on_AF_ex(af_ex_framework):
    """
    Tests the analytical scores on the AF_ex.af file.
    This test confirms the main calculation logic against the complex example.
    The expected scores are derived from the logic in the ProbAdmissible class,
    assuming the graph structure from the thesis (p=0.5).
    Score(a) = Pr(a exists) * Pr(a is conflict-free) * Pr(a is defended)
    """
    # 1. Initialize the semantics calculator with the framework and p=0.5
    prob = ProbAdmissible(af_ex_framework, p=0.5)

    # 2. Get the calculated scores
    calculated_scores = prob.get_scores()

    # 3. Define the "golden" expected scores for AF_ex with p=0.5
    # These scores are based on an analytical derivation of Pr({arg} is admissible)
    expected_scores = {
        '1': 0.5,       # Unattacked: score = p
        '2': 0.25,      # Attacked by '1': score = p * (1-p)
        '3': 0.125,     # Attacked by '2', '6': score = p * (1-p) * (1-p)
        '4': 0.125,     # Attacked by '1', '7': score = p * (1-p) * (1-p)
        '5': 0.25,      # Attacked by '1': score = p * (1-p)
        '6': 0.5,       # Unattacked: score = p
        '7': 0.25,      # Attacked by '8': score = p * (1-p)
        '8': 0.125      # Attacked by '4', '5': score = p * (1-p) * (1-p)
    }

    # 4. Assert that the calculated dictionary matches the expected one.
    # pytest.approx is used for safe floating-point comparison.
    assert calculated_scores == pytest.approx(expected_scores)

def test_prob_admissible_single_argument(single_arg_framework):
    """
    Tests the scoring on a framework with a single, unattacked argument.
    """
    prob = ProbAdmissible(single_arg_framework, p=0.5)
    scores = prob.get_scores()
    expected = {'1': 0.5} # Pr({'1'} is admissible) = Pr('1' exists) = p
    assert scores == pytest.approx(expected)

def test_prob_admissible_simple_attack(simple_attack_framework):
    """
    Tests the scoring on a framework with a simple attack (1 -> 2).
    """
    prob = ProbAdmissible(simple_attack_framework, p=0.5)
    scores = prob.get_scores()
    expected = {
        '1': 0.5,   # Unattacked: score = p
        '2': 0.25   # Attacked by 1, not defended: score = p * (1-p)
    }
    assert scores == pytest.approx(expected)

def test_prob_admissible_self_attack(self_attack_framework):
    """
    Tests that a self-attacking argument has a score of 0, as it can never
    be in a conflict-free set.
    """
    prob = ProbAdmissible(self_attack_framework, p=0.5)
    scores = prob.get_scores()
    expected = {'1': 0.0} # Pr({'1'} is conflict-free) = 0
    assert scores == pytest.approx(expected)

def test_prob_admissible_mutual_attack(mutual_attack_framework):
    """
    Tests a mutual attack (1 <-> 2). Each argument can defend itself.
    """
    prob = ProbAdmissible(mutual_attack_framework, p=0.5)
    scores = prob.get_scores()
    # For '1', it is attacked by '2'. '1' defends itself by attacking '2'.
    # Pr('1' is defended from '2') = Pr('2' doesn't exist) + Pr('2' exists AND '1' attacks '2')
    # = (1-p) + p*1 = 1.
    # Score('1') = Pr('1' exists) * Pr('1' defended) = p * 1 = 0.5
    expected = {
        '1': 0.5,
        '2': 0.5
    }
    assert scores == pytest.approx(expected)