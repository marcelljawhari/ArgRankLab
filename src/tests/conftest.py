# tests/conftest.py

import pytest
import os
import networkx as nx
from util.af_parser import parse_af_file

@pytest.fixture(scope="module")
def af_ex_framework():
    """A pytest fixture that loads the AF_ex framework once per test module."""
    af_file_path = os.path.join("data", "AF_ex.af")
    graph = parse_af_file(af_file_path)
    return graph

@pytest.fixture
def single_arg_framework():
    """Creates a very simple framework with only one argument '1'."""
    graph = nx.DiGraph()
    graph.add_node("1")
    return graph

@pytest.fixture
def simple_attack_framework():
    """Creates a simple framework with one attack: 1 -> 2."""
    graph = nx.DiGraph()
    graph.add_edge("1", "2")
    return graph


@pytest.fixture
def mutual_attack_framework():
    """Creates a framework with two mutually attacking arguments."""
    graph = nx.DiGraph()
    graph.add_edge("1", "2")
    graph.add_edge("2", "1")
    return graph

@pytest.fixture
def three_cycle_framework():
    """Creates a framework with a 3-argument cycle (1->2, 2->3, 3->1)."""
    graph = nx.DiGraph()
    graph.add_edge("1", "2")
    graph.add_edge("2", "3")
    graph.add_edge("3", "1")
    return graph

@pytest.fixture
def self_attack_framework():
    """Creates a framework with a single self-attacking argument."""
    graph = nx.DiGraph()
    graph.add_edge("1", "1")
    return graph

@pytest.fixture
def defense_chain_framework():
    """Creates a simple defense chain: 1 -> 2 -> 3."""
    graph = nx.DiGraph()
    graph.add_edge("1", "2")
    graph.add_edge("2", "3")
    return graph

@pytest.fixture
def split_defense_framework():
    """
    Creates a framework where two arguments defend a fourth:
    1->3, 2->3, 3<->4.
    The preferred extensions are {1,4} and {2,4}.
    The intersection is {4}, so the ideal extension is {4}.
    """
    graph = nx.DiGraph()
    graph.add_edge("1", "3")
    graph.add_edge("2", "3")
    graph.add_edge("3", "4")
    graph.add_edge("4", "3")
    return graph
