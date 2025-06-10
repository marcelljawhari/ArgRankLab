# test/test_dbs.py
import os
from util.af_parser import parse_af_file
from semantics.dbs import Dbs

def test_dbs_on_AF_ex(af_ex_framework):
    """
    Tests the Dbs ranking on the AF_ex.af file using pytest.
    """
    print("\n--- Running Dbs test on AF_ex.af from file ---")

    # 1. Calculate the ranking
    dbs = Dbs(af_ex_framework, max_path_length=5)

    # 2. Get the actual ranking and format it for comparison
    calculated_ranking = [sorted(list(group)) for group in dbs.get_ranking()]

    # 3. Define the expected result
    expected_ranking = [
        ['1', '6'], ['7'], ['2', '5'], ['8'], ['3'], ['4']
    ]

    # 4. Assert that the result matches the expectation
    assert calculated_ranking == expected_ranking