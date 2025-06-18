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
