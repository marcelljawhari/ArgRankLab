# tests/conftest.py

import pytest
import os
from util.af_parser import parse_af_file

@pytest.fixture(scope="module")
def af_ex_framework():
    """A pytest fixture that loads the AF_ex framework once per test module."""
    af_file_path = os.path.join("data", "AF_ex.af")
    graph = parse_af_file(af_file_path)
    return graph